
# python pca_test.py \
#     --model "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image" \
#     --dataset blip3o \
#     --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets\
#     --num-samples 10 \
#     --timestep-idx 25 \
#     --vis-sample-size 100000 \
#     --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/vis/origin_t25_test0 \
#     --dataset-train \

#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen-image
cd /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual


python text_token_analysis.py \
    --model "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image" \
    --dataset blip3o \
    --datadir /inspire/hdd/project/chineseculture/public/yuxuan/datasets\
    --num-samples 10 \
    --timestep-idx 25 \
    --vis-sample-size 100000 \
    --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/vis/base_t25_normalize_LN \
    --dataset-train \
    --normalize-layers \
    



# python text_token_analysis.py \
#     --model "/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image" \
#     --timestep-idx 40 \
#     --vis-sample-size 1000 \
#     --output-dir /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/output/vis/test-t40 \
#     --prompt "A warm, sunlit living room with a minimalist and cozy aesthetic. A wooden coffee table stands in the center, holding a white ceramic mug, a small potted succulent, and a neat stack of books. Soft natural light pours through the window, casting gentle shadows across the table and the textured beige rug below. In the background sits a comfortable beige sofa with fabric upholstery. The scene is calm, tidy, and atmospheric, emphasizing warm tones, natural materials, and a serene morning mood." \
#     --image "/inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/test_resized.jpg" \