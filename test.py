
from diffusers import QwenImagePipeline
from sampler import MyQwenImagePipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

model_name = "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
# pipe = MyQwenImagePipeline.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     residual_origin_layer=0,
#     residual_target_layers=[15, 20],
#     residual_weights=[0.0, 0.0]
# ).to("cuda")
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "default": (1024, 1024),
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["default"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
)

# out = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     width=width,
#     height=height,
#     num_inference_steps=50,
#     true_cfg_scale=4.0,
#     generator=torch.Generator(device="cuda").manual_seed(42),
#     collect_layers=[1, 2, 3],
#     target_timestep=25
# )
# image = out['images'][0]
# text_tokens = out['text_layer_outputs']
# feats0 = out["origin0"]
# feats1 = out["origin1"]
# feats2 = out["text_layer_outputs"][0][-1]

image.save("example-test.png")

