#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#    Qwen-Image Text Token Analysis (CKNNA / Cos / PCA)
#    - Uses MyQwenImagePipeline.__call__ with collect_layers
#    - No manual VAE encode / pack / scheduling
#    - Uses target_timestep to extract transformer text features
# ================================================================

import argparse
import math
import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sampler import MyQwenImagePipeline
from datasets import get_target_dataset
from torch.utils.data import ConcatDataset, Subset


# ================================================================
# Metrics
# ================================================================
def cknna(feats_a: torch.Tensor, feats_b: torch.Tensor, topk: int = 10, unbiased: bool = False) -> float:
    if feats_a.shape[0] <= 1:
        return float("nan")

    feats_a = torch.nn.functional.normalize(feats_a.float(), dim=-1)
    feats_b = torch.nn.functional.normalize(feats_b.float(), dim=-1)

    sim_a = feats_a @ feats_a.t()
    sim_b = feats_b @ feats_b.t()

    diag_mask = torch.eye(feats_a.shape[0], dtype=torch.bool, device=feats_a.device)
    sim_a = sim_a.masked_fill(diag_mask, -float("inf"))
    sim_b = sim_b.masked_fill(diag_mask, -float("inf"))

    k = max(1, min(topk, feats_a.shape[0] - 1))
    nn_a = torch.topk(sim_a, k=k, dim=-1).indices
    nn_b = torch.topk(sim_b, k=k, dim=-1).indices

    overlaps = []
    for i in range(feats_a.shape[0]):
        sa = set(nn_a[i].tolist())
        sb = set(nn_b[i].tolist())
        common = len(sa.intersection(sb))

        if unbiased:
            expected = (k * k) / float(feats_a.shape[0] - 1)
            denom = max(1e-6, k - expected)
            common = max(0.0, common - expected) / denom
        else:
            common = common / float(k)

        overlaps.append(common)

    return float(np.mean(overlaps))


def cosine_mean(a: torch.Tensor, b: torch.Tensor):
    if a.shape[0] == 0:
        return float("nan"), np.array([])
    sim = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    return sim.mean().item(), sim.cpu().numpy()


# ================================================================
# Utils
# ================================================================
def load_and_resize(src, height: int, width: int):
    if isinstance(src, Image.Image):
        img = src.convert("RGB")

    elif torch.is_tensor(src):
        t = src.detach().cpu()
        if t.dim() == 4 and t.shape[0] == 1:
            t = t[0]
        if t.dim() != 3:
            raise ValueError(f"Unexpected tensor shape: {t.shape}")
        if t.min() < 0 or t.max() > 1:
            t = (t + 1) / 2
            t = t.clamp(0, 1)
        from torchvision.transforms import ToPILImage
        img = ToPILImage()(t)

    elif isinstance(src, str):
        img = Image.open(src).convert("RGB")

    else:
        raise TypeError(f"Unsupported input: {type(src)}")

    return img.resize((width, height), Image.BICUBIC)


def pil_to_tensor(img: Image.Image, device: torch.device):
    from torchvision.transforms import ToTensor
    return ToTensor()(img).unsqueeze(0).to(device)


# ================================================================
# PCA
# ================================================================
def fit_pca(all_features):
    from sklearn.decomposition import PCA
    valid = [f for f in all_features if f is not None and f.shape[0] > 2]
    if not valid:
        return None
    concat = np.concatenate(valid, axis=0)
    if concat.shape[0] < 3:
        return None
    return PCA(n_components=2, random_state=42).fit(concat)


def add_subplot_border(ax, color="gray", lw: float = 1.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def plot_pca(layer_tokens, pca, outdir, layers):
    if pca is None:
        print("[WARN] PCA skipped, insufficient data.")
        return

    n = len(layers)
    nc = min(6, n)
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr))
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        ax = axes[i]
        feats = layer_tokens[i]
        if feats is not None and feats.shape[0] > 2:
            pts = pca.transform(feats)
            split = 77 if pts.shape[0] >= 333 else min(77, pts.shape[0])
            if split > 0:
                ax.scatter(pts[:split, 0], pts[:split, 1], s=6, alpha=0.3, color="royalblue")
            if pts.shape[0] - split > 0:
                ax.scatter(pts[split:, 0], pts[split:, 1], s=6, alpha=0.3, color="tomato")
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        add_subplot_border(ax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    outfile = os.path.join(outdir, "text_emb_pca.png")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {outfile}")


def plot_curves(results, outpath):
    xs = [r[0] for r in results]
    cvals = [r[1] for r in results]
    covals = [r[2] for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(xs, cvals, marker="o", label="CKNNA")
    plt.plot(xs, covals, marker="s", linestyle="--", label="Cosine")
    plt.grid(alpha=0.3)
    plt.xlabel("Layer")
    plt.ylabel("Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ================================================================
# Dataset
# ================================================================
def prepare_dataset(args):
    if not args.dataset:
        if args.prompt is None or args.image is None:
            raise ValueError("No dataset: need --image and --prompt.")
        return None, 1

    if args.datadir is None:
        raise ValueError("--datadir required for dataset.")

    datasets = [
        get_target_dataset(name, args.datadir, train=args.dataset_train, transform=None)
        for name in args.dataset
    ]
    dset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total = len(dset)
    if args.num_samples > 0 and args.num_samples < total:
        dset = Subset(dset, list(range(args.num_samples)))
        total = args.num_samples

    print(f"[DATASET] Loaded {total} samples")
    return dset, total


def extract_pair(sample):
    if isinstance(sample, dict):
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        txt = sample.get("prompt") or sample.get("caption") or sample.get("text")
        return img, txt
    if isinstance(sample, (tuple, list)):
        return sample[0], sample[1]
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def iterate_pairs(dataset, total, args):
    if dataset is None:
        yield 0, args.image, args.prompt
    else:
        for i in range(total):
            img, txt = extract_pair(dataset[i])
            yield i, img, txt


# ================================================================
# Main
# ================================================================
def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("[INIT] Loading pipeline...")
    pipe = MyQwenImagePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    dataset, total = prepare_dataset(args)

    target_layers = sorted(set(args.layers))
    all_layers = [0] + target_layers

    # Storage
    layer0_all = []
    layerX_all = {l: [] for l in target_layers}
    cknna_vals = {l: [] for l in target_layers}
    cosine_vals = {l: [] for l in target_layers}
    token_lengths = []

    os.makedirs(args.output_dir, exist_ok=True)

    # Process
    for idx, img_in, prompt in iterate_pairs(dataset, total, args):
        print(f"\n=== [{idx+1}/{total}] Prompt: {prompt[:60]} ===")


        # Run full official pipeline; collect text features
        out = pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=1024,
            height=1024,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            collect_layers=target_layers,
            target_timestep=args.timestep_idx,
            num_inference_steps=args.num_inference_steps,
        )

        txt_dict = out["text_layer_outputs"]

        # Layer 0
        layer0 = txt_dict[0][-1].float()   # last = target timestep
        layer0_all.append(layer0.cpu())
        token_lengths.append(layer0.shape[0])

        # Other layers
        for l in target_layers:
            feats = txt_dict[l][-1].float()  # target_timestep
            layerX_all[l].append(feats.cpu())

            ckn = cknna(feats, layer0, topk=args.topk, unbiased=args.unbiased)
            co, _ = cosine_mean(feats, layer0)
            cknna_vals[l].append(ckn)
            cosine_vals[l].append(co)

    # Summary
    results = []
    for l in target_layers:
        mean_ck = float(np.mean(cknna_vals[l]))
        mean_co = float(np.mean(cosine_vals[l]))
        results.append((l, mean_ck, mean_co))
        print(f"Layer {l}: CKNNA={mean_ck:.4f}, Cos={mean_co:.4f}")

    plot_curves(results, os.path.join(args.output_dir, args.output_name))

    # PCA
    layer_tokens_np = []
    for l in all_layers:
        toks = layer0_all if l == 0 else layerX_all[l]
        if len(toks) == 0:
            layer_tokens_np.append(None)
        else:
            feats = torch.cat(toks, dim=0)
            if args.vis_sample_size > 0 and feats.shape[0] > args.vis_sample_size:
                feats = feats[torch.randperm(feats.shape[0])[:args.vis_sample_size]]
            layer_tokens_np.append(feats.numpy())

    pca = fit_pca(layer_tokens_np)
    plot_pca(layer_tokens_np, pca, args.output_dir, all_layers)


# ================================================================
# Args
# ================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--dataset", type=str, nargs="+", default=None)
    p.add_argument("--datadir", type=str, default=None)
    p.add_argument("--dataset-train", action="store_true")
    p.add_argument("--num-samples", type=int, default=-1)

    p.add_argument("--timestep-idx", type=int, required=True)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(1, 60)))
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--unbiased", action="store_true")

    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="attn_vis_out")
    p.add_argument("--output-name", type=str, default="cknna_cosine_curve.png")
    p.add_argument("--vis-sample-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
