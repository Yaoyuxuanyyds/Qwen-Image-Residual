import argparse
import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid, save_image

from sampler import MyQwenImagePipeline


torch.set_grad_enabled(False)


# ============================================================
# Qwen-Image Generator（替代 SD3ImageGenerator）
# ============================================================
class QwenImageGenerator:
    def __init__(
        self,
        model_dir,
        residual_target_layers=None,
        residual_origin_layer=None,
        residual_weights=None,
        true_cfg_scale=4.0,
        num_inference_steps=50,
        width=1024,
        height=1024,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ----- load pipeline -----
        self.pipe = MyQwenImagePipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            residual_origin_layer=residual_origin_layer,
            residual_target_layers=residual_target_layers,
            residual_weights=residual_weights,
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        # 生成参数
        self.true_cfg_scale = true_cfg_scale
        self.num_inference_steps = num_inference_steps
        self.width = width
        self.height = height

    # --------------------------------------------------------
    # generate single image
    # --------------------------------------------------------
    def generate_image(self, prompt, seed, **kwargs):
        g = torch.Generator(device=self.device).manual_seed(seed)
        negative_prompt = " "

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.true_cfg_scale,
            generator=g,
        ).images[0]

        # PIL → tensor
        return torch.from_numpy(np.array(out)).permute(2, 0, 1) / 255.0



# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_grid", action="store_true")

    # 生成参数
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    # residual fused skip connection
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)

    # DPG-style multi-GPU
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)

    return parser.parse_args()



# ============================================================
# 主流程
# ============================================================
def main(opt):
    # ===== metadata 加载 =====
    if not os.path.exists(opt.metadata_file):
        raise FileNotFoundError(f"metadata 文件不存在：{opt.metadata_file}")

    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    total_items = len(metadatas)

    # ===== 手动分片 =====
    if opt.world_size > 1:
        local_indices = [i for i in range(total_items) if i % opt.world_size == opt.rank]
    else:
        local_indices = list(range(total_items))

    print(f"[Rank {opt.rank}] 总任务 {total_items}，本卡处理 {len(local_indices)}")

    # ===== 初始化 QwenImage 生成器 =====
    generator = QwenImageGenerator(
        model_dir=opt.model_dir,
        residual_target_layers=opt.residual_target_layers,
        residual_origin_layer=opt.residual_origin_layer,
        residual_weights=opt.residual_weights,
        true_cfg_scale=opt.cfg,
        num_inference_steps=opt.steps,
        width=opt.width,
        height=opt.height,
    )

    # ===== 遍历当前 rank 的任务 =====
    for index in local_indices:
        metadata = metadatas[index]
        prompt = metadata["prompt"]

        outpath = os.path.join(opt.outdir, f"{index:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        print(f"[Rank {opt.rank}] Prompt {index:05d}: {prompt}")

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        all_samples = []

        # 多次采样
        for _ in trange(
            (opt.n_samples + opt.batch_size - 1) // opt.batch_size,
            desc=f"[Rank {opt.rank}] Sampling {index:05d}",
        ):
            cur_bs = min(opt.batch_size, opt.n_samples - sample_count)

            for _ in range(cur_bs):
                img_path = os.path.join(sample_path, f"{sample_count:05d}.png")

                if os.path.exists(img_path):
                    print(f"[skip] 存在: {img_path}")
                    sample_count += 1
                    continue

                image = generator.generate_image(
                    prompt=prompt,
                    seed=opt.seed + sample_count,
                )

                save_image(image, img_path, normalize=True)

                if not opt.skip_grid:
                    all_samples.append(image.unsqueeze(0))

                sample_count += 1

        # grid 保存
        if not opt.skip_grid and all_samples:
            grid = torch.cat(all_samples, dim=0)
            grid = make_grid(grid, nrow=min(opt.batch_size, opt.n_samples))
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(
                os.path.join(outpath, "grid.png")
            )

        del all_samples

    print(f"[Rank {opt.rank}] 完成！")



# ============================================================
if __name__ == "__main__":
    opt = parse_args()
    main(opt)
