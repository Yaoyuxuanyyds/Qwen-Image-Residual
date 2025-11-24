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
    "$(seq -s ' ' 3 44)"
)

RES_WEIGHT_LIST=(
    "$(printf '0.5 %.0s' $(seq 3 44))"
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

    EXP_NAME="target-${EXP_TARGET_SHORT}__origin-${RES_ORIGIN}__w-${EXP_WEIGHT_SHORT}"




    SAVEDIR="${BASE_SAVE_DIR}/${EXP_NAME}"
    GENEVAL_OUTDIR="${BASE_GENEVAL_DIR}/${EXP_NAME}"
    DPG_OUTDIR="${DPG_SAVE_BASE}/${EXP_NAME}"
    T2I_OUTDIR="${BASE_T2I_DIR}/${EXP_NAME}"

    # mkdir -p "$SAVEDIR" "$GENEVAL_OUTDIR" "$DPG_OUTDIR" "$T2I_OUTDIR"

    # echo "→ SAVEDIR:        $SAVEDIR"
    # echo "→ GENEVAL_OUTDIR: $GENEVAL_OUTDIR"
    # echo "→ DPG_OUTDIR:     $DPG_OUTDIR"
    # echo "→ T2I_OUTDIR:     $T2I_OUTDIR"


    # # ① Geneval 多卡并行生成
    # echo "📌 Running GenEval bench generation (multi-GPU)..."

    # WORLD_SIZE=8   # 你要用的 GPU 数量（可改成你自己的量）

    # for RANK in $(seq 0 $((WORLD_SIZE-1))); do
    #     CUDA_VISIBLE_DEVICES=$RANK python generate_geneval.py \
    #         --seed 42 \
    #         --batch_size $BATCHSIZE \
    #         --model_dir $MODEL_DIR \
    #         --metadata_file /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval/prompts/evaluation_metadata.jsonl \
    #         --outdir "$GENEVAL_OUTDIR" \
    #         --residual_target_layers $RES_TARGET \
    #         --residual_origin_layer $RES_ORIGIN \
    #         --residual_weight $RES_WEIGHT \
    #         --world_size $WORLD_SIZE \
    #         --rank $RANK \
    #         --skip_grid \
    #         > "${GENEVAL_OUTDIR}/log_rank${RANK}.txt" 2>&1 &
    # done

    # wait
    # echo "🎉 GenEval multi-GPU generation finished!"



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
    # echo "🎉 T2I generation finished."

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












# ============================================================
# =============== 阶段 2：Geneval 测评 =======================
# ============================================================
echo "============================================"
echo " Phase 2: Running Geneval evaluation "
echo "============================================"

conda activate geneval_1
cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/geneval

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





# # ============================================================
# # =============== 阶段 3：sample.py 结果测评 =================
# # ============================================================
# echo "============================================"
# echo " Phase 3: Evaluating sample.py generated images "
# echo "============================================"

# # ⚠ 回到 repa-sd3 环境运行 eval.py
# conda activate repa-sd3
# cd /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3

# for SAVEDIR in "${SAMPLE_DIR_LIST[@]}"; do

#     echo "----------------------------------------------------"
#     echo "Running eval.py for:"
#     echo "    $SAVEDIR"
#     echo "----------------------------------------------------"

#     python eval.py \
#         --load_dir "$SAVEDIR" \
#         --datadir "$DATADIR" \
#         --load_name "${DATASET}-cfg${CFG}-nfe${NFE}" \
#         --benchmark $BENCHMARKS \
#         --num $NUM_SAMPLES

#     echo "🎉 eval.py finished for: $SAVEDIR"
#     echo
# done


# # ============================================================
# # =============== 阶段 4：DPG Bench 测评 =============
# # ============================================================
# echo "============================================"
# echo " Phase 4: Running DPG Bench evaluation (official) "
# echo "============================================"

# DPG_BENCH_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench"
# DPG_RESOLUTION=1024   # 单格尺寸，官方要求


# cd /inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench
# source /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3-1/ELLA/.venv/bin/activate


# # 为每个 residual 实验进行 DPG 测评
# for DPG_OUTDIR in "${DPG_DIR_LIST[@]}"; do

#     DPG_EVAL_RES="${DPG_SAVE_BASE}/results/${EXP_NAME}.txt"

#     echo "----------------------------------------------------"
#     echo " Evaluating DPG directory: $DPG_OUTDIR"
#     echo "----------------------------------------------------"

#     python compute_dpg_bench.py \
#         --image-root-path "$DPG_OUTDIR" \
#         --res-path "$DPG_EVAL_RES" \
#         --resolution $DPG_RESOLUTION

#     echo "DPG evaluation finished: $DPG_OUTDIR"
#     echo "    → Log file: "$DPG_EVAL_RES"
#     echo
# done

# echo "🎉 All DPG Bench evaluations completed!"
