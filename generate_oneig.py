import argparse
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from sampler import MyQwenImagePipeline
import megfile


torch.set_grad_enabled(False)


# ===========================
# Helper
# ===========================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    img_tensor: [3, H, W], values in [0,1] or arbitrary → clamp 到 [0,1] 再转 PIL
    """
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    arr = (img_tensor * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def create_image_gallery(images, rows=2, cols=2):
    """
    images: PIL.Image 列表，至少 rows*cols 张
    返回拼好的大图（rows × cols）
    """
    assert len(images) >= rows * cols, "Not enough images provided!"

    img_w, img_h = images[0].size  # PIL: (width, height)

    gallery_w = cols * img_w
    gallery_h = rows * img_h
    gallery_image = Image.new("RGB", (gallery_w, gallery_h))

    for r in range(rows):
        for c in range(cols):
            img = images[r * cols + c]
            x_offset = c * img_w
            y_offset = r * img_h
            gallery_image.paste(img, (x_offset, y_offset))

    return gallery_image


# category → 子目录名
CLASS_ITEM = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


# ===========================
# Qwen-Image 封装
# ===========================
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

    def generate(self, prompt: str, seed: int) -> torch.Tensor:
        """
        返回 [3, H, W]、值在 [0,1] 左右的 tensor
        """
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
        return arr


# ===========================
# Args
# ===========================
def parse_args():
    parser = argparse.ArgumentParser()

    # Qwen-Image 相关
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--img_size", type=int, default=1024)

    # residual fusion
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    # OneIG-Bench / 存储配置
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "cn"],
        help="使用英文 prompt_en 还是中文 prompt_cn",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="顶层输出目录（例如 images）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen_image",
        help="子目录中的模型名，例如 images/<class>/<model_name>/id.webp",
    )

    # grid 设置
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=2)

    # 多卡
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()


# ===========================
# Main
# ===========================
def main(opt):
    set_seed(opt.seed)

    # ====== 初始化 Qwen-Image ======
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

    # ====== 读取 CSV ======
    if opt.lang == "en":
        csv_path =  "/inspire/hdd/project/chineseculture/public/yuxuan/benches/OneIG-Benchmark/OneIG-Bench.csv"
    else:
        csv_path =  "/inspire/hdd/project/chineseculture/public/yuxuan/benches/OneIG-Benchmark/OneIG-Bench-ZH.csv"
    df = pd.read_csv(csv_path, dtype=str)
    total_rows = len(df)

    # 多卡切分：按 index % world_size
    if opt.world_size > 1:
        local_indices = [i for i in range(total_rows) if i % opt.world_size == opt.rank]
    else:
        local_indices = list(range(total_rows))

    print(
        f"[Qwen-OneIG] world_size={opt.world_size}, rank={opt.rank}, "
        f"total_rows={total_rows}, this_rank_rows={len(local_indices)}"
    )

    total_slots = opt.grid_rows * opt.grid_cols

    # ====== 遍历本 rank 负责的样本 ======
    for idx in local_indices:
        row = df.iloc[idx]
        category = row["category"]
        sample_id = row["id"]

        # 选择 prompt 语言
        if opt.lang == "en":
            prompt = row["prompt_en"]
        else:
            prompt = row["prompt_cn"]
        

        print(f"\n[Rank {opt.rank}] idx={idx}, id={sample_id}, cat={category}")
        print(f"Prompt: {prompt}")

        # 如果类别不在 mapping 中，可以选择跳过或创建默认目录
        if category not in CLASS_ITEM:
            print(f"[Rank {opt.rank}] Warning: unknown category {category}, skip.")
            continue

        # ====== 生成 images (tensor→PIL) ======
        images = []
        for i in range(total_slots):
            img_tensor = generator.generate(prompt=prompt, seed=opt.seed + i)
            images.append(tensor_to_pil(img_tensor))

        # ====== 防御：不足则用黑图填 ======
        if len(images) == 0:
            black = Image.new("RGB", (opt.img_size, opt.img_size), color=(0, 0, 0))
            images = [black] * total_slots
        elif len(images) < total_slots:
            img_w, img_h = images[0].size
            black = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
            images.extend([black] * (total_slots - len(images)))

        # ====== 拼 gallery ======
        gallery = create_image_gallery(images, rows=opt.grid_rows, cols=opt.grid_cols)

        # ====== 构造保存路径（megfile）=====
        subfolder = CLASS_ITEM[category]
        dir_path = megfile.smart_path_join(opt.image_dir, subfolder, opt.model_name)
        file_path = megfile.smart_path_join(dir_path, f"{sample_id}.webp")

        # 有的后端可能会自动建目录；稳妥一点可以：
        try:
            megfile.smart_makedirs(dir_path, exist_ok=True)
        except Exception:
            # 如果后端不支持 makedirs 就算了，交给 smart_open 处理
            pass

        with megfile.smart_open(file_path, "wb") as f:
            gallery.save(f, format="WEBP")

        print(f"[Rank {opt.rank}] Saved: {file_path}")

    print(f"\n[Rank {opt.rank}] All done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
