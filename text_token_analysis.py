#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
#    Qwen-Image Text Token Analysis (CKNNA / Cos / PCA / UMAP / t-SNE)
#    - Uses MyQwenImagePipeline.__call__ with collect_layers
#    - No manual VAE encode / pack / scheduling
#    - Uses target_timestep to extract transformer text features
#    - PCA / UMAP / t-SNE all fitted on concatenated features of ALL layers
# ================================================================

import argparse
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


def fix_feat_shape(f: torch.Tensor) -> torch.Tensor:
    """
    将 [1, T, D] 或 [B, T, D] 变成 [T, D]。
    """
    if f.dim() == 3:
        if f.shape[0] == 1:
            return f.squeeze(0)   # [1,T,D] → [T,D]
        else:
            return f.reshape(-1, f.shape[-1])  # [B,T,D] → [B*T,D]
    return f


# ================================================================
# PCA & global embedding helpers
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


def plot_pca(layer_tokens, pca, outdir, layers, layer_stats):
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
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.3, color="tomato")

        # ---- NEW: annotate stats ----
        if layer in layer_stats:
            m = layer_stats[layer]["mean"]
            s = layer_stats[layer]["std"]
            ax.text(
                0.01, 0.99,
                f"μ={m:.2f}\nσ={s:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2)
            )

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


# >>> 新增：把所有层的特征拼成一个矩阵，并记录每层在全局矩阵中的区间
def build_global_matrix(layer_tokens_np: List[Optional[np.ndarray]]) -> Tuple[Optional[np.ndarray], List[Tuple[int,int]]]:
    """
    返回：
      X_all: [N_all, D]
      spans: len == num_layers, 每个元素是 (start, end)，用于从 X_all / embedding 中取出对应层数据
    """
    arrays = []
    spans: List[Tuple[int, int]] = []
    start = 0
    for arr in layer_tokens_np:
        if arr is None or arr.shape[0] == 0:
            spans.append((start, start))
            continue
        n = arr.shape[0]
        arrays.append(arr)
        end = start + n
        spans.append((start, end))
        start = end

    if not arrays:
        return None, spans

    X_all = np.concatenate(arrays, axis=0)
    return X_all, spans


# >>> 新增：UMAP / t-SNE 拟合（在全局矩阵上）
def fit_umap_global(X: Optional[np.ndarray]):
    if X is None or X.shape[0] < 3:
        print("[WARN] UMAP skipped, insufficient data.")
        return None
    try:
        import umap
    except ImportError:
        print("[WARN] umap-learn not installed; skip UMAP.")
        return None

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    emb = reducer.fit_transform(X)
    return emb


def fit_tsne_global(X: Optional[np.ndarray]):
    if X is None or X.shape[0] < 3:
        print("[WARN] t-SNE skipped, insufficient data.")
        return None
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN] sklearn.manifold.TSNE not available; skip t-SNE.")
        return None

    # 注意：大样本时 t-SNE 很慢，这里只做 2D 可视化用途
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    emb = tsne.fit_transform(X)
    return emb


# >>> 新增：根据全局 embedding + spans，按层画 UMAP / t-SNE
def plot_embedding_per_layer(
    emb: Optional[np.ndarray],
    spans: List[Tuple[int, int]],
    layers: Sequence[int],
    outdir: str,
    tag: str = "umap",
):
    """
    emb: [N_all,2]，是所有层共同空间的 2D 嵌入
    spans[i] = (start,end)，表示第 i 个 layer 在 emb 中的 slice
    """
    if emb is None:
        return

    n = len(layers)
    nc = min(6, n)
    nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr))
    axes = axes.flatten()

    for i, layer in enumerate(layers):
        ax = axes[i]
        start, end = spans[i]
        coords = emb[start:end]
        if coords.shape[0] > 0:
            ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.3, color="tomato")
        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        add_subplot_border(ax)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    outfile = os.path.join(outdir, f"text_emb_{tag}.png")
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {outfile}")


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

    os.makedirs(args.output_dir, exist_ok=True)

    # Process
    for idx, img_in, prompt in iterate_pairs(dataset, total, args):
        print(f"\n=== [{idx+1}/{total}] Prompt: {prompt[:60]} ===")

        # 这里只用 prompt，不使用 img_in（走标准 T2I 推理）
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + idx)

        out = pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=args.width,
            height=args.height,
            true_cfg_scale=4.0,
            generator=gen,
            collect_layers=target_layers,
            target_timestep=args.timestep_idx,
            num_inference_steps=args.num_inference_steps,
        )

        txt_dict = out["text_layer_outputs"]

        # Layer 0
        layer0 = fix_feat_shape(txt_dict[0][-1].float())
        layer0_all.append(layer0.cpu())

        # Other layers
        for l in target_layers:
            feats = fix_feat_shape(txt_dict[l][-1].float())
            layerX_all[l].append(feats.cpu())

            ckn = cknna(feats, layer0, topk=args.topk, unbiased=args.unbiased)
            co, _ = cosine_mean(feats, layer0)
            cknna_vals[l].append(ckn)
            cosine_vals[l].append(co)

    # Summary
    results = []
    for l in target_layers:
        mean_ck = float(np.mean(cknna_vals[l])) if cknna_vals[l] else float("nan")
        mean_co = float(np.mean(cosine_vals[l])) if cosine_vals[l] else float("nan")
        results.append((l, mean_ck, mean_co))
        print(f"Layer {l}: CKNNA={mean_ck:.4f}, Cos={mean_co:.4f}")

    plot_curves(results, os.path.join(args.output_dir, args.output_name))

    # ====== 构建各层特征（下采样后） ======
    layer_stats = {}
    layer_tokens_np: List[Optional[np.ndarray]] = []
    for l in all_layers:
        toks = layer0_all if l == 0 else layerX_all[l]
        if len(toks) == 0:
            layer_tokens_np.append(None)
        else:
            feats = torch.cat(toks, dim=0)

            # ======================================================
            # NEW: record pre-normalization stats (scalar mean/std)
            # ======================================================
            pre_mean_vec = feats.mean(-1, keepdims=True).cpu().numpy()   # shape [1, D]
            pre_std_vec  = feats.std(-1, keepdims=True).cpu().numpy()    # shape [1, D]
            layer_stats[l] = {
                "mean": float(pre_mean_vec.mean()),     # scalar mean
                "std": float(pre_std_vec.mean()),       # scalar std
            }

            # ======================================================
            # optional per-layer normalization
            # ======================================================
            if args.normalize_layers:
                feats = feats - feats.mean(-1, keepdims=True)
                feats = feats / (feats.std(-1, keepdims=True) + 1e-6)

                
            if args.vis_sample_size > 0 and feats.shape[0] > args.vis_sample_size:
                feats = feats[torch.randperm(feats.shape[0])[:args.vis_sample_size]]
            layer_tokens_np.append(feats.numpy())

    # ====== PCA（全局拟合，原逻辑保留） ======
    pca = fit_pca(layer_tokens_np)
    plot_pca(layer_tokens_np, pca, args.output_dir, all_layers, layer_stats)

    # ================================================================
    # 新增：layer0 tokens 单独 PCA（只对 layer0 拟合）
    # ================================================================
    print("[INFO] Fitting PCA ONLY on layer0 tokens (single-layer PCA)...")

    # layer0 的特征在 layer_tokens_np[0]
    layer0_np = layer_tokens_np[0]

    if layer0_np is None or layer0_np.shape[0] < 3:
        print("[WARN] layer0 tokens insufficient for PCA, skip.")
    else:
        from sklearn.decomposition import PCA

        pca0 = PCA(n_components=2, random_state=42).fit(layer0_np)
        pts0 = pca0.transform(layer0_np)

        # ---- 绘图 ----
        fig = plt.figure(figsize=(5, 5))
        plt.scatter(pts0[:, 0], pts0[:, 1], s=6, alpha=0.35, color="steelblue")
        plt.title("Layer0 Only PCA", fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.grid(alpha=0.2)

        out_png = os.path.join(args.output_dir, "text_emb_layer0_pca.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

        print(f"[SAVE] {out_png}")


    # # ====== UMAP / t-SNE：在所有层拼接的共同空间中拟合，然后按层可视化 ======
    # X_all, spans = build_global_matrix(layer_tokens_np)

    # umap_emb = fit_umap_global(X_all)
    # plot_embedding_per_layer(umap_emb, spans, all_layers, args.output_dir, tag="umap")

    # tsne_emb = fit_tsne_global(X_all)
    # plot_embedding_per_layer(tsne_emb, spans, all_layers, args.output_dir, tag="tsne")


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
    p.add_argument("--normalize-layers", action="store_true")
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
