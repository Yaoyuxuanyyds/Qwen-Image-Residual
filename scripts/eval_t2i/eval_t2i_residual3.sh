#!/bin/bash

# source /inspire/hdd/project/chineseculture/yaoyuxuan-CZXS25220085/p-yaoyuxuan/REPA-SD3-1/T2I-CompBench/.venv/bin/activate
# # 清除pip约束相关环境变量（核心！解决constraint.txt错误）
# unset PIP_CONSTRAINT  # 强制取消pip的约束文件设置
# unset PIP_CONFIG_FILE  # 临时禁用pip配置文件（避免读取全局配置）
# # 2. 锁定路径：只保留虚拟环境+基础命令路径
# export PATH="$VIRTUAL_ENV/bin:/bin:/usr/bin"
# export PYTHONPATH="$VENV_SITE_PACKAGES"  # 核心：只加载虚拟环境的库
# export LD_LIBRARY_PATH="$PYTORCH_LIB_PATH"

# # 3. 强制设置CUDA_VERSION=11.8
# export CUDA_VERSION=11.8


cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench

# ============================================================
# =============== 参数列表（可自由扩展） =====================
# ============================================================
RES_ORIGIN_LIST=(1)

RES_TARGET_LIST=(
    "$(seq -s ' ' 3 59)"
)

RES_WEIGHT_LIST=(
    "$(printf '0.1 %.0s' $(seq 3 59))"
)


# =============== 输出目录配置 ===============================
BASE_T2I_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/output-qwenimage"
BASE_MODEL="qwen_residual"

T2I_DIR_LIST=()  


# ============================================================
# =============== 阶段 1：生成（sample + geneval + DPG）=====
# ============================================================
for RES_TARGET in "${RES_TARGET_LIST[@]}"; do
for RES_ORIGIN in "${RES_ORIGIN_LIST[@]}"; do
for RES_WEIGHT in "${RES_WEIGHT_LIST[@]}"; do

    echo "====================================================="
    echo "🔍 Running residual experiment:"
    echo "  → residual_target_layers : ${RES_TARGET}"
    echo "  → residual_origin_layer  : ${RES_ORIGIN}"
    echo "  → residual_weight        : ${RES_WEIGHT}"
    echo "====================================================="

    # 自动压缩 target layers 显示形式
    FIRST_LAYER=$(echo "$RES_TARGET" | awk '{print $1}')
    LAST_LAYER=$(echo "$RES_TARGET" | awk '{print $NF}')
    EXP_TARGET_SHORT="${FIRST_LAYER}to${LAST_LAYER}"

    # 权重统一就取第一个即可
    FIRST_WEIGHT=$(echo "$RES_WEIGHT" | awk '{print $1}')
    EXP_WEIGHT_SHORT="${FIRST_WEIGHT}"

    EXP_NAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}"




    T2I_OUTDIR="${BASE_T2I_DIR}/${EXP_NAME}"
    echo "→ T2I_OUTDIR:     $T2I_OUTDIR"

    cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/UniDet_eval
    python 2D_spatial_eval_new.py --outpath "$T2I_OUTDIR" --sample_subdir "${BASE_MODEL}_spatial_val"
    python 3D_spatial_eval_new.py --outpath "$T2I_OUTDIR" --sample_subdir "${BASE_MODEL}_3d_spatial_val"
    python numeracy_eval_new.py --outpath "$T2I_OUTDIR" --sample_subdir "${BASE_MODEL}_numeracy_val"

    cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/BLIPvqa_eval
    python BLIP_vqa_new.py   --out_dir "$T2I_OUTDIR" --sample_subdir  "${BASE_MODEL}_shape_val"
    python BLIP_vqa_new.py   --out_dir "$T2I_OUTDIR" --sample_subdir  "${BASE_MODEL}_color_val"
    python BLIP_vqa_new.py   --out_dir "$T2I_OUTDIR" --sample_subdir  "${BASE_MODEL}_texture_val"

    cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/CLIPScore_eval
    python CLIP_similarity_new.py --outpath "$T2I_OUTDIR" --sample_subdir "${BASE_MODEL}_non_spatial_val" 

    cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench
    python all_scores_simple.py \
        --sample_subdir "${BASE_MODEL}" \
        --examples_root  "$T2I_OUTDIR" 

done
done
done

echo "🎉🎉 All residual experiments completed!"
echo










