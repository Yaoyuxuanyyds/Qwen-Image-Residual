#!/bin/bash
set -euo pipefail




# ============================================================
# =============== 参数列表（可自由扩展） =====================
# ============================================================
RES_ORIGIN_LIST=(1)

RES_TARGET_LIST=(
    "$(seq -s ' ' 3 44)"
)

RES_WEIGHT_LIST=(
    "$(printf '0.1 %.0s' $(seq 3 44))"
)



# =============== 输出目录配置 ===============================
DPG_SAVE_BASE="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/outputs-qwenimage"
DPG_DIR_LIST=()    



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

    DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"

    # 保存目录列表用于后续 Stage
    DPG_DIR_LIST+=("$DPG_OUTDIR")


done
done
done

echo "🎉🎉 All residual experiments completed!"
echo







# ============================================================
# =============== 阶段 4：DPG Bench 测评 =============
# ============================================================
echo "============================================"
echo " Phase 4: Running DPG Bench evaluation (official) "
echo "============================================"

DPG_BENCH_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench"
DPG_RESOLUTION=1024   # 单格尺寸，官方要求


# cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench
# source /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/ELLA/.venv/bin/activate


# 为每个 residual 实验进行 DPG 测评
for DPG_OUTDIR in "${DPG_DIR_LIST[@]}"; do

    DPG_EVAL_RES="${DPG_SAVE_BASE}/results/${EXP_NAME}.txt"

    echo "----------------------------------------------------"
    echo " Evaluating DPG directory: $DPG_OUTDIR"
    echo "----------------------------------------------------"

    python compute_dpg_bench.py \
        --image-root-path "$DPG_OUTDIR" \
        --res-path "$DPG_EVAL_RES" \
        --resolution $DPG_RESOLUTION

    echo "DPG evaluation finished: $DPG_OUTDIR"
    echo "    → Log file: "$DPG_EVAL_RES" "
    echo
done

echo "🎉 All DPG Bench evaluations completed!"
