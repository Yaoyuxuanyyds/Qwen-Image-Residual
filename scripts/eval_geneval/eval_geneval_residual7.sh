#!/bin/bash
set -euo pipefail

source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate geneval_1
cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval



# ============================================================
# =============== 参数列表（可自由扩展） =====================
# ============================================================
RES_ORIGIN_LIST=(31)

RES_TARGET_LIST=(
    "$(seq -s ' ' 32 44)"
)

RES_WEIGHT_LIST=(
    "$(printf '0.25 %.0s' $(seq 32 44))"
)




# =============== 输出目录配置 ===============================
BASE_GENEVAL_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs-qwenimage/residual_eval"

GENEVAL_DIR_LIST=()


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


    GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"

    # 保存目录列表用于后续 Stage
    GENEVAL_DIR_LIST+=("$GENEVAL_OUTDIR")
done
done
done

echo "🎉🎉 All residual experiments completed!"
echo











# ============================================================
# =============== 阶段 2：Geneval 测评 =======================
# ============================================================
echo "============================================"
echo " Phase 2: Running Geneval evaluation "
echo "============================================"


MASK2FORMER_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/mask2former"

for GENEVAL_OUTDIR in "${GENEVAL_DIR_LIST[@]}"; do
    echo "----------------------------------------------------"
    echo " Evaluating Geneval directory:"
    echo "   $GENEVAL_OUTDIR"
    echo "----------------------------------------------------"

    STEP_NAME=$(basename "$GENEVAL_OUTDIR")
    OUTFILE_PARENT=$(dirname "$GENEVAL_OUTDIR")
    GENEVAL_OUTFILE="${OUTFILE_PARENT}/results_${STEP_NAME}.jsonl"

    python evaluation/evaluate_images.py \
        "$GENEVAL_OUTDIR" \
        --outfile "$GENEVAL_OUTFILE" \
        --model-path "$MASK2FORMER_PATH"

    python evaluation/summary_scores.py \
        "$GENEVAL_OUTFILE"

    echo "🎉 Geneval evaluation finished: $STEP_NAME"
    echo
done
