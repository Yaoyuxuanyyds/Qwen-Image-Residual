#!/bin/bash
set -euo pipefail

# =============== 阶段 0：环境 ===============
source /inspire/hdd/project/chineseculture/public/yuxuan/miniconda3/etc/profile.d/conda.sh
conda activate qwen-image
cd /inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual

MODEL='qwen-image'
MODEL_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/Qwen-Image"

NFE=50
CFG=4.0
IMGSIZE=1024
DATASET="coco"
BENCHMARKS="ImageReward-v1.0,CLIP,PickScore,FID,LPIPS"
NUM_SAMPLES=-1
DATADIR="/inspire/hdd/project/chineseculture/public/yuxuan/datasets"
BATCHSIZE=16




# ============================================================
# =============== 参数列表（可自由扩展） =====================
# ============================================================
RES_ORIGIN_LIST=(1)

RES_TARGET_LIST=(
    "$(seq -s ' ' 2 11)"
)

RES_WEIGHT_LIST=(
    "$(printf '0.1 %.0s' $(seq 2 11))"
)


# =============== 输出目录配置 ===============================
BASE_SAVE_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/Qwen-Image-Residual/logs/residual_eval"
BASE_GENEVAL_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/outputs-qwenimage/residual_eval"
BASE_T2I_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/T2I-CompBench/output-qwenimage"
DPG_SAVE_BASE="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/outputs-qwenimage"
mkdir -p "$BASE_SAVE_DIR" "$BASE_GENEVAL_DIR" "$BASE_T2I_DIR" "$DPG_SAVE_BASE"


GENEVAL_DIR_LIST=()
SAMPLE_DIR_LIST=()
DPG_DIR_LIST=()    
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

    EXP_NAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}-LayerNorm"

    SAVEDIR="${BASE_SAVE_DIR}/${EXP_NAME}"
    GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"
    DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"
    T2I_OUTDIR="${BASE_T2I_DIR}/${EXP_NAME}"

    mkdir -p "$SAVEDIR" "$GENEVAL_OUTDIR" "$DPG_OUTDIR" "$T2I_OUTDIR"

    echo "→ SAVEDIR:        $SAVEDIR"
    echo "→ GENEVAL_OUTDIR: $GENEVAL_OUTDIR"
    echo "→ DPG_OUTDIR:     $DPG_OUTDIR"
    echo "→ T2I_OUTDIR:     $T2I_OUTDIR"


    # ① Geneval 多卡并行生成
    echo "📌 Running GenEval bench generation (multi-GPU)..."

    WORLD_SIZE=8   # 你要用的 GPU 数量（可改成你自己的量）

    for RANK in $(seq 0 $((WORLD_SIZE-1))); do
        CUDA_VISIBLE_DEVICES=$RANK python generate_geneval.py \
            --seed 42 \
            --batch_size $BATCHSIZE \
            --model_dir $MODEL_DIR \
            --metadata_file /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/prompts/evaluation_metadata.jsonl \
            --outdir "$GENEVAL_OUTDIR" \
            --residual_target_layers $RES_TARGET \
            --residual_origin_layer $RES_ORIGIN \
            --residual_weight $RES_WEIGHT \
            --world_size $WORLD_SIZE \
            --rank $RANK \
            --skip_grid \
            > "${GENEVAL_OUTDIR}/log_rank${RANK}.txt" 2>&1 &
    done

    wait
    echo "🎉 GenEval multi-GPU generation finished!"



    echo "📌 Running DPG bench generation on 8 GPUs..."
    WORLD_SIZE=8
    for GPU_ID in $(seq 0 $((WORLD_SIZE-1))); do
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate_dpg.py \
            --save_dir "$DPG_OUTDIR" \
            --img_size $IMGSIZE \
            --residual_target_layers $RES_TARGET \
            --residual_origin_layer $RES_ORIGIN \
            --residual_weight $RES_WEIGHT \
            --world_size $WORLD_SIZE \
            --rank $GPU_ID \
            > "${DPG_OUTDIR}/log_gpu${GPU_ID}.txt" 2>&1 &
    done

    wait    # <-- 必须等待所有并行任务完成
    echo "🎉 DPG multi-GPU generation finished!"



    # echo "📌 Running Multi-GPU Generation..."
    # WORLD_SIZE=8
    # for GPU_ID in $(seq 0 $((WORLD_SIZE-1))); do
    #     CUDA_VISIBLE_DEVICES=$GPU_ID python generate_t2i.py \
    #         --outdir_base "${T2I_OUTDIR}" \
    #         --output_prefix "qwen_residual" \
    #         --residual_target_layers $RES_TARGET \
    #         --residual_origin_layer $RES_ORIGIN \
    #         --residual_weight $RES_WEIGHT \
    #         --world_size $WORLD_SIZE \
    #         --rank $GPU_ID \
    #         > "${T2I_OUTDIR}/log_gpu${GPU_ID}.txt" 2>&1 &
    # done

    # wait
    # echo "🎉 T2I multi-GPU generation finished."

    # # # sample.py 生成图片
    # # echo "📌 Running Basic bench generation..."
    # # python sample.py \
    # #     --cfg_scale $CFG --NFE $NFE --model $MODEL --img_size $IMGSIZE --batch_size $BATCHSIZE \
    # #     --save_dir "$SAVEDIR" --datadir "$DATADIR" --num $NUM_SAMPLES --dataset "$DATASET" \
    # #     --residual_target_layers $RES_TARGET \
    # #     --residual_origin_layer $RES_ORIGIN \
    # #     --residual_weight $RES_WEIGHT



    # 保存目录列表用于后续 Stage
    GENEVAL_DIR_LIST+=("$GENEVAL_OUTDIR")
    SAMPLE_DIR_LIST+=("$SAVEDIR")
    DPG_DIR_LIST+=("$DPG_OUTDIR")
    T2I_DIR_LIST+=("$T2I_OUTDIR")

done
done
done

echo "🎉🎉 All residual experiments completed!"
echo




