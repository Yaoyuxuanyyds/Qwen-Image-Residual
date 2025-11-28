#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#   Qwen-Image Text Token Geometry Analysis
#   - Dataset mode
#   - Collect origin0 / origin1 / layer0 from MyQwenImagePipeline
#   - Independent PCA for each group
#   - Output a 1x3 subplot: origin0 | origin1 | layer0
# ================================================================

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, Subset
from PIL import Image

from sampler import MyQwenImagePipeline
from datasets import get_target_dataset


# ================================================================
# Helper
# ================================================================
def fix_feat_shape(f: torch.Tensor) -> torch.Tensor:
    """
    Convert [1,T,D] → [T,D] or [B,T,D] → [B*T,D]
    """
    if f.dim() == 3:
        if f.shape[0] == 1:
            return f.squeeze(0)
        return f.reshape(-1, f.shape[-1])
    return f


def concat_and_sample(feat_list, max_n):
    if len(feat_list) == 0:
        return None
    feats = torch.cat(feat_list, dim=0)   # [N, D]
    if max_n > 0 and feats.shape[0] > max_n:
        feats = feats[torch.randperm(feats.shape[0])[:max_n]]
    return feats.cpu().numpy()


# ================================================================
# Dataset loader
# ================================================================
def prepare_dataset(args):
    if args.dataset is None:
        raise ValueError("This version requires --dataset mode.")

    if args.datadir is None:
        raise ValueError("--datadir required.")

    datasets = [
        get_target_dataset(name, args.datadir, train=args.dataset_train, transform=None)
        for name in args.dataset
    ]
    dset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    total = len(dset)

    if args.num_samples > 0 and args.num_samples < total:
        dset = Subset(dset, list(range(args.num_samples)))
        total = args.num_samples

    print(f"[DATASET] Loaded {total} samples.")
    return dset, total


def extract_pair(sample):
    if isinstance(sample, dict):
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        txt = sample.get("prompt") or sample.get("caption") or sample.get("text")
        return img, txt
    if isinstance(sample, (tuple, list)):
        return sample[0], sample[1]
    raise TypeError(f"Unsupported sample type: {type(sample)}")


# ================================================================
# Main
# ================================================================
def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load pipeline
    print("[INIT] Loading pipeline...")
    pipe = MyQwenImagePipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    dataset, total = prepare_dataset(args)

    # Storage
    origin0_list = []
    origin1_list = []
    layer0_list = []

    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate dataset
    for idx in range(total):
        img, prompt = extract_pair(dataset[idx])
        print(f"[{idx+1}/{total}] Prompt: {str(prompt)[:70]}")

        gen = torch.Generator(device=device).manual_seed(args.seed + idx)

        out = pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=args.width,
            height=args.height,
            true_cfg_scale=4.0,
            num_inference_steps=args.num_inference_steps,
            generator=gen,
            collect_layers=[1,2,3],        # 只需要 layer0 的 txt_feats
            target_timestep=args.timestep_idx,
        )

        # store
        origin0_list.append(fix_feat_shape(out["origin0"].float()).cpu())
        origin1_list.append(fix_feat_shape(out["origin1"].float()).cpu())
        layer0 = fix_feat_shape(out["text_layer_outputs"][0][-1].float())
        layer0_list.append(layer0.cpu())

    # ================================================================
    # Build PCA inputs
    # ================================================================
    f0 = concat_and_sample(origin0_list, args.vis_sample_size)
    f1 = concat_and_sample(origin1_list, args.vis_sample_size)
    f2 = concat_and_sample(layer0_list, args.vis_sample_size)

    # ================================================================
    # Fit independent PCA
    # ================================================================
    from sklearn.decomposition import PCA

    def run_pca(feats):
        if feats is None or feats.shape[0] < 3:
            return None
        pca = PCA(n_components=2, random_state=42).fit(feats)
        return pca.transform(feats)

    p0 = run_pca(f0)
    p1 = run_pca(f1)
    p2 = run_pca(f2)

    # ================================================================
    # Plot 1x3 subplots
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = ["origin0 (raw hidden)", "origin1 (RMSNorm)", "layer0 (RMSNorm + Linear)"]
    plist = [p0, p1, p2]

    for ax, pts, title in zip(axes, plist, names):
        if pts is not None:
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, color="tomato")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)

    fig.tight_layout()
    outfile = os.path.join(args.output_dir, "pca_origin0_origin1_layer0.png")
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"[SAVE] {outfile}")


# ================================================================
# Args
# ================================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, nargs="+", required=True)
    p.add_argument("--datadir", type=str, required=True)
    p.add_argument("--dataset-train", action="store_true")
    p.add_argument("--num-samples", type=int, default=64)

    p.add_argument("--timestep-idx", type=int, required=True)
    p.add_argument("--num-inference-steps", type=int, default=50)

    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="token_vis")
    p.add_argument("--vis-sample-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
