import argparse
import numpy as np
import random, os, glob
import torch
from torchvision.utils import save_image
from PIL import Image
from einops import rearrange

from sampler import MyQwenImagePipeline


torch.set_grad_enabled(False)


# =============================================================================
# Helper
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_grid_2x2(imgs):
    """imgs: list of 4 tensors, each [3,H,W], values in [0,1]."""
    assert len(imgs) == 4

    pil_imgs = [(img.clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy()
                for img in imgs]
    pil_imgs = [Image.fromarray(p) for p in pil_imgs]

    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    grid.paste(pil_imgs[0], (0, 0))
    grid.paste(pil_imgs[1], (w, 0))
    grid.paste(pil_imgs[2], (0, h))
    grid.paste(pil_imgs[3], (w, h))

    return grid


# =============================================================================
# Qwen-Image Generator (简化版)
# =============================================================================

class QwenImageGenerator:
    def __init__(
        self,
        model_dir,
        true_cfg_scale=4.0,
        num_inference_steps=50,
        width=1024,
        height=1024,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
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

        self.true_cfg_scale = true_cfg_scale
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height

    def generate(self, prompt, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=" ",
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.true_cfg_scale,
            generator=g,
        ).images[0]

        arr = torch.from_numpy(np.array(out)).permute(2, 0, 1) / 255.0
        return arr  # tensor [3,H,W]


# =============================================================================
# Args
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--img_size", type=int, default=1024)

    parser.add_argument("--model_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/prompts")

    # residual fusion
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    # 多卡
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main(opt):
    set_seed(opt.seed)

    os.makedirs(opt.save_dir, exist_ok=True)

    # ===== Qwen-Image 生成器 =====
    generator = QwenImageGenerator(
        model_dir=opt.model_dir,
        true_cfg_scale=opt.cfg_scale,
        num_inference_steps=opt.steps,
        width=opt.img_size,
        height=opt.img_size,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
    )

    # ===== 扫描 prompts =====
    txt_files = sorted(glob.glob(os.path.join(opt.prompt_dir, "*.txt")))
    total_prompts = len(txt_files)

    # ===== DPG 风格多卡分片 =====
    if opt.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % opt.world_size == opt.rank]

    print(f"[Qwen-Image DPG] world_size={opt.world_size}, rank={opt.rank}")
    print(f"[Qwen-Image DPG] total prompts={total_prompts}, this rank={len(txt_files)}")

    # ===== 遍历每个 prompt =====
    for txt_path in txt_files:
        name = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(opt.save_dir, f"{name}.png")

        if os.path.exists(out_path):
            print(f"[Skip] {name}.png already exists")
            continue

        # 读取 prompt
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\n[Rank {opt.rank}] Generating {name} ...")

        # 生成 4 张图 ⇒ 拼成 2×2 grid
        imgs = []
        for i in range(4):
            img = generator.generate(prompt=prompt, seed=opt.seed + i)
            imgs.append(img)

        grid = make_grid_2x2(imgs)
        grid.save(out_path)

        print(f"[Saved] {out_path}")

    print(f"\n[Rank {opt.rank}] All done!")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
