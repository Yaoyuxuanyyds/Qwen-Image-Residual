#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Layer-wise text token similarity & PCA visualization for Qwen-Image.

This script mirrors the SD3 reference workflow: it collects per-layer text
token embeddings from ``MyQwenImageTransformer2DModel`` (via
``target_layers``), computes cosine and CKNNA similarities against the layer-0
embeddings, plots the curves, and produces a PCA visualization that is fitted
across all layers' tokens together.
"""

import argparse
import json
import math
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sampler import MyQwenImagePipeline


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def cknna(feats_a: torch.Tensor, feats_b: torch.Tensor, topk: int = 10, unbiased: bool = False) -> float:
    """Cross-kNN agreement between two embedding sets.

    For each token, we compute the k nearest neighbors (cosine similarity) in
    both spaces and measure the overlap ratio. ``unbiased`` follows the common
    correction that subtracts the expected random overlap.
    """

    if feats_a.shape[0] <= 1:
        return float("nan")

    feats_a = torch.nn.functional.normalize(feats_a.float(), dim=-1)
    feats_b = torch.nn.functional.normalize(feats_b.float(), dim=-1)

    sim_a = feats_a @ feats_a.t()
    sim_b = feats_b @ feats_b.t()

    # Exclude self
    diag_mask = torch.eye(feats_a.shape[0], dtype=torch.bool, device=feats_a.device)
    sim_a = sim_a.masked_fill(diag_mask, -float("inf"))
    sim_b = sim_b.masked_fill(diag_mask, -float("inf"))

    k = max(1, min(topk, feats_a.shape[0] - 1))
    nn_a = torch.topk(sim_a, k=k, dim=-1).indices
    nn_b = torch.topk(sim_b, k=k, dim=-1).indices

    overlaps = []
    for idx in range(feats_a.shape[0]):
        set_a = set(nn_a[idx].tolist())
        set_b = set(nn_b[idx].tolist())
        common = len(set_a.intersection(set_b))
        if unbiased:
            # Expected random overlap under independence
            expected = (k * k) / float(feats_a.shape[0] - 1)
            denom = max(1e-6, k - expected)
            common = max(0.0, common - expected) / denom
        else:
            common = common / float(k)
        overlaps.append(common)

    return float(np.mean(overlaps))


def cosine_mean(feats_a: torch.Tensor, feats_b: torch.Tensor) -> Tuple[float, np.ndarray]:
    if feats_a.shape[0] == 0:
        return float("nan"), np.array([])
    sim = torch.nn.functional.cosine_similarity(feats_a.float(), feats_b.float(), dim=-1)
    return sim.mean().item(), sim.cpu().numpy()


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def load_and_resize(image_path: str, height: int, width: int) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img.resize((width, height), Image.BICUBIC)


def pil_to_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    import torchvision.transforms as T

    t = T.ToTensor()
    return t(pil_img).unsqueeze(0).to(device)


def encode_image(pipe: MyQwenImagePipeline, img_tensor: torch.Tensor) -> torch.Tensor:
    vae = pipe.vae
    scaling = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    with torch.no_grad():
        posterior = vae.encode(img_tensor * 2 - 1)
        latents = posterior.latent_dist.sample()
        latents = (latents - shift) * scaling
    return latents


def add_noise(pipe: MyQwenImagePipeline, latents: torch.Tensor, timestep_idx: int, num_inference_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps)
    if timestep_idx < 0 or timestep_idx >= len(scheduler.timesteps):
        raise ValueError(f"timestep_idx must be in [0, {len(scheduler.timesteps) - 1}]")
    timestep = scheduler.timesteps[timestep_idx].to(latents.device)
    noise = torch.randn_like(latents)
    noisy = scheduler.add_noise(latents, noise, timestep)
    return noisy, timestep


# -----------------------------------------------------------------------------
# Prompt helpers
# -----------------------------------------------------------------------------
def encode_prompt(pipe: MyQwenImagePipeline, prompt: str, device: torch.device):
    if hasattr(pipe, "encode_prompt"):
        outputs = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        if isinstance(outputs, (list, tuple)):
            prompt_embeds = outputs[0]
            attention_mask = outputs[-1] if len(outputs) > 1 else None
        elif isinstance(outputs, dict):
            prompt_embeds = outputs.get("prompt_embeds")
            attention_mask = outputs.get("attention_mask")
        else:
            prompt_embeds = outputs
            attention_mask = None
    elif hasattr(pipe, "_encode_prompt"):
        prompt_embeds, _, attention_mask = pipe._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    else:
        raise AttributeError("Pipeline does not provide encode_prompt or _encode_prompt")

    return prompt_embeds, attention_mask


# -----------------------------------------------------------------------------
# PCA helpers
# -----------------------------------------------------------------------------
def fit_pca(all_features: List[np.ndarray]):
    from sklearn.decomposition import PCA

    valid = [f for f in all_features if f is not None and f.shape[0] > 2]
    if not valid:
        return None
    concat = np.concatenate(valid, axis=0)
    if concat.shape[0] < 3:
        return None
    return PCA(n_components=2, random_state=42).fit(concat)


def add_subplot_border(ax, color="gray", lw=1.0):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------
def process_pair(
    pipe: MyQwenImagePipeline,
    image_path: str,
    prompt: str,
    args: argparse.Namespace,
) -> Tuple[dict, List[torch.Tensor], List[torch.Tensor]]:
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else torch.device(args.device)

    pil_img = load_and_resize(image_path, args.height, args.width)
    img_tensor = pil_to_tensor(pil_img, device=device)
    latents = encode_image(pipe, img_tensor)
    noisy_latents, timestep = add_noise(pipe, latents, args.timestep_idx, args.num_inference_steps)

    prompt_embeds, attention_mask = encode_prompt(pipe, prompt, device)

    # img_shapes: (height, width, patch_size) for rotary embedding; txt_seq_lens: token length
    img_shapes = [(noisy_latents.shape[2], noisy_latents.shape[3], pipe.transformer.config.patch_size)]
    txt_seq_lens = [prompt_embeds.shape[1]]

    outputs = pipe.transformer(
        hidden_states=noisy_latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=attention_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        target_layers=args.layers,
    )

    return outputs, outputs["txt_feats_list"], outputs["context_embedder_output"]


def parse_pairs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.pairs_json is None:
        if args.prompt is None or args.image is None:
            raise ValueError("Either --pairs-json or both --prompt/--image must be provided")
        return [(args.image, args.prompt)]

    with open(args.pairs_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    pairs = []
    for item in records:
        pairs.append((item["image"], item["prompt"]))
    return pairs


def plot_curves(layer_results: List[Tuple[int, float, float]], output_path: str):
    xs = [r[0] for r in layer_results]
    cknna_vals = [r[1] for r in layer_results]
    cosine_vals = [r[2] for r in layer_results]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, cknna_vals, marker="o", label="CKNNA")
    plt.plot(xs, cosine_vals, marker="s", linestyle="--", label="Mean Cosine")
    plt.xlabel("Layer index")
    plt.ylabel("Similarity vs layer 0")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pca(layer_tokens: List[Optional[np.ndarray]], pca, output_dir: str, layers: Sequence[int]):
    if pca is None:
        print("[WARN] Not enough tokens for PCA; skipping plot")
        return

    n_layers = len(layers)
    ncols = min(6, n_layers)
    nrows = int(math.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        feats = layer_tokens[idx]
        if feats is not None and feats.shape[0] > 2:
            pts = pca.transform(feats)
            split = 77 if pts.shape[0] >= 333 else min(77, pts.shape[0])
            if split > 0:
                ax.scatter(pts[:split, 0], pts[:split, 1], s=8, alpha=0.3, color="royalblue", label="CLIP tokens")
            if pts.shape[0] - split > 0:
                ax.scatter(pts[split:, 0], pts[split:, 1], s=8, alpha=0.3, color="tomato", label="T5 tokens")
            if pts.shape[0] > 2:
                ax.legend(fontsize=6, loc="best", frameon=False)
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        add_subplot_border(ax)

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PCA (global fit) of text token embeddings across layers", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pca = os.path.join(output_dir, "text_emb_pca.png")
    fig.savefig(out_pca, dpi=200)
    plt.close(fig)
    print(f"[SAVE] PCA plot saved to {out_pca}")


def run(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pipe = MyQwenImagePipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    pairs = parse_pairs(args)
    target_layers = sorted(set(args.layers))
    all_layers = [0] + target_layers

    os.makedirs(args.output_dir, exist_ok=True)

    layer_cknna_records = {l: [] for l in target_layers}
    layer_cos_records = {l: [] for l in target_layers}
    layer_token_collections = {l: [] for l in all_layers}
    token_counts = []

    for idx, (image_path, prompt) in enumerate(pairs):
        print(f"[PAIR {idx + 1}/{len(pairs)}] {os.path.basename(image_path)} | prompt preview: {prompt[:60]}")
        outputs, txt_feats_list, context_embedder_output = process_pair(pipe, image_path, prompt, args)

        layer0_feats = context_embedder_output[0][0].detach().to(torch.float32)
        layer_token_collections[0].append(layer0_feats.cpu())
        token_counts.append(layer0_feats.shape[0])

        for layer, feats in zip(target_layers, txt_feats_list):
            feats_single = feats[0].detach().to(torch.float32)
            layer_token_collections[layer].append(feats_single.cpu())
            cknna_val = cknna(feats_single, layer0_feats, topk=args.topk, unbiased=args.unbiased)
            cosine_val, _ = cosine_mean(feats_single, layer0_feats)
            layer_cknna_records[layer].append(cknna_val)
            layer_cos_records[layer].append(cosine_val)

    if len(token_counts) == 0:
        print("[WARN] No valid pairs processed.")
        return

    print(f"[STATS] Processed {len(pairs)} pairs. Avg valid tokens: {float(np.mean(token_counts)):.2f}")

    results = []
    for layer in target_layers:
        mean_cknna = float(np.mean(layer_cknna_records[layer])) if layer_cknna_records[layer] else float("nan")
        mean_cosine = float(np.mean(layer_cos_records[layer])) if layer_cos_records[layer] else float("nan")
        results.append((layer, mean_cknna, mean_cosine))
        print(f"Layer {layer:02d}: mean CKNNA={mean_cknna:.6f}, mean Cosine={mean_cosine:.6f}")

    plot_curves(results, os.path.join(args.output_dir, args.output_name))
    print(f"[SAVE] Curve plot saved to {os.path.join(args.output_dir, args.output_name)}")

    # PCA
    layer_feats_np: List[Optional[np.ndarray]] = []
    for layer in all_layers:
        tokens = layer_token_collections[layer]
        if len(tokens) == 0:
            layer_feats_np.append(None)
            continue
        feats = torch.cat(tokens, dim=0)
        if args.num_samples != 1 and args.vis_sample_size > 0 and feats.shape[0] > args.vis_sample_size:
            perm = torch.randperm(feats.shape[0])[: args.vis_sample_size]
            feats = feats[perm]
        layer_feats_np.append(feats.numpy())

    pca = fit_pca(layer_feats_np)
    plot_pca(layer_feats_np, pca, args.output_dir, all_layers)


def parse_args():
    p = argparse.ArgumentParser(description="Compute CKNNA & cosine similarities across Qwen-Image layers.")
    p.add_argument("--model", type=str, required=True, help="Path or model id for the Qwen-Image pipeline")
    p.add_argument("--prompt", type=str, default=None, help="Prompt text when evaluating a single pair")
    p.add_argument("--image", type=str, default=None, help="Image path when evaluating a single pair")
    p.add_argument("--pairs-json", dest="pairs_json", type=str, default=None, help="JSON list of {image, prompt} for batched evaluation")
    p.add_argument("--timestep-idx", dest="timestep_idx", type=int, required=True)
    p.add_argument("--num-inference-steps", dest="num_inference_steps", type=int, default=50)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(1, 23)))
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="attn_vis_out")
    p.add_argument("--output-name", type=str, default="cknna_cosine_curve.png")
    p.add_argument("--vis-sample-size", dest="vis_sample_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", dest="num_samples", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    run(args)

