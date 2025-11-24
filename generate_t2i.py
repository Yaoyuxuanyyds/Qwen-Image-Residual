import argparse
import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

from sampler import MyQwenImagePipeline

torch.set_grad_enabled(False)
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
# ===============================================================
# Qwen-Image 封装（替代 SD3ImageGenerator）
# ===============================================================

class QwenImageGenerator:
    def __init__(
        self,
        model_dir,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        cfg=4.0,
        steps=50,
        width=1024,
        height=1024,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe = MyQwenImagePipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            residual_origin_layer=residual_origin_layer,
            residual_target_layers=residual_target_layers,
            residual_weights=residual_weights,
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        self.cfg = cfg
        self.steps = steps
        self.width = width
        self.height = height

    def generate(self, prompt, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=self.width,
            height=self.height,
            num_inference_steps=self.steps,
            true_cfg_scale=self.cfg,
            generator=g,
        ).images[0]

        arr = torch.from_numpy(np.array(out)).permute(2, 0, 1) / 255.0
        return arr  # tensor [3,H,W]


# ===============================================================
# 工具
# ===============================================================

def clean_prompt_for_filename(prompt):
    illegal_chars = r'[\/:*?"<>|]'
    cleaned = re.sub(illegal_chars, "_", prompt.strip())
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000] + "_truncated"
    return cleaned


# ===============================================================
# 参数
# ===============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/T2I-CompBench/examples/dataset")
    parser.add_argument("--model_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image")
    parser.add_argument("--outdir_base", type=str, required=True)

    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627],
    )

    parser.add_argument("--output_prefix", type=str, default="qwen_t2i")

    # residual
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    # 多卡
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    opt = parser.parse_args()
    assert len(opt.seeds) >= opt.n_samples, "Seeds 数量不足 n_samples"

    return opt


# ===============================================================
# 主流程（多卡分片）
# ===============================================================

def main(opt):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Rank {opt.rank}] Device={device}, world_size={opt.world_size}")

    generator = QwenImageGenerator(
        model_dir=opt.model_dir,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        cfg=opt.cfg,
        steps=opt.steps,
        width=opt.width,
        height=opt.height,
    )

    # 扫描 *val.txt
    txt_files = sorted(glob.glob(os.path.join(opt.dataset_dir, "*val.txt")))
    total_files = len(txt_files)

    # 多卡分片
    if opt.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % opt.world_size == opt.rank]

    print(f"[Rank {opt.rank}] total txt={total_files}, this rank={len(txt_files)}")

    seeds_to_use = opt.seeds[:opt.n_samples]

    # 遍历 txt 文件
    for txt_path in txt_files:
        txt_name = os.path.splitext(os.path.basename(txt_path))[0]
        outdir = os.path.join(opt.outdir_base, f"samples_{opt.output_prefix}_{txt_name}")
        os.makedirs(outdir, exist_ok=True)

        # 加载 prompts
        with open(txt_path, "r") as f:
            prompts = [x.strip() for x in f if x.strip()]

        # 遍历每行 prompt
        for prompt in tqdm(prompts, desc=f"[Rank {opt.rank}] {txt_name}"):
            prompt_clean = clean_prompt_for_filename(prompt)

            for si, seed in enumerate(seeds_to_use):
                fname = f"{prompt_clean}_{si:06d}.png"
                fpath = os.path.join(outdir, fname)

                if os.path.exists(fpath):
                    continue

                img = generator.generate(prompt=prompt, seed=seed)
                save_image(img, fpath, normalize=True)

    print(f"[Rank {opt.rank}] All Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
