#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: $0 BUDGET SEED [--pe true|false] [--idct true|false]"
  echo "Defaults: PE=true; IDCT=false for 4/9/16 tokens and true for 25 tokens. Basis: DCT."
  exit 0
fi
source "$(dirname "$0")/common.sh"

parse_braco_run "$@"
set_repro_seed "${SEED}"

GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
if [[ ! "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "GRAD_ACCUM_STEPS must be a positive integer" >&2
  exit 2
fi

REPO="${UPSTREAM_ROOT}/llava"
STAGE1="${RUN_ROOT}/stage1"
STAGE2="${RUN_ROOT}/stage2"
echo "Braco: K=${BUDGET}, basis=DCT, PE=${USE_PE}, IDCT=${LOW_IDCT}; output=${RUN_ROOT}"

require_dir "${REPO}"
require_dir "${MODEL_ROOT}/vicuna-7b-v1.5"
require_dir "${MODEL_ROOT}/clip-vit-large-patch14-336"
require_file "${DATA_ROOT}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
require_file "${DATA_ROOT}/LLaVA-Instruct/llava_v1_5_mix665k.json"
record_environment "${RUN_ROOT}/metadata"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

deepspeed llava/train/train_mem.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path "${MODEL_ROOT}/vicuna-7b-v1.5" \
  --version plain \
  --data_path "${DATA_ROOT}/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
  --image_folder "${DATA_ROOT}/LLaVA-Pretrain/images" \
  --vision_tower "${MODEL_ROOT}/clip-vit-large-patch14-336" \
  --mm_projector_type fourier_mlp2x_gelu \
  --mm_fourier_C "${C}" \
  --mm_fourier_K_high 0 \
  --mm_fourier_spatial_keep "${S}" \
  --mm_fourier_temp_start 2.0 \
  --mm_fourier_temp_end 1.2 \
  --mm_fourier_temp_steps 2180 \
  --mm_fourier_spatial_selection_norm sparsemax \
  --mm_fourier_spatial_hidden_mult 1 \
  --mm_fourier_spatial_scale_init 1.0 \
  --mm_fourier_spatial_prior_weight_init 0.0 \
  --mm_fourier_spatial_use_norm True \
  --mm_fourier_use_polar_pe "${USE_PE}" \
  --mm_fourier_polar_pe_scale_init 0.1 \
  --mm_fourier_post_concat_norm False \
  --mm_fourier_low_idct "${LOW_IDCT}" \
  --tune_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir "${STAGE1}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 1e-3 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --seed "${SEED}" \
  --data_seed "${SEED}"

require_file "${STAGE1}/mm_projector.bin"

deepspeed llava/train/train_mem.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path "${MODEL_ROOT}/vicuna-7b-v1.5" \
  --version v1 \
  --data_path "${DATA_ROOT}/LLaVA-Instruct/llava_v1_5_mix665k.json" \
  --image_folder "${DATA_ROOT}/LLaVA-Instruct/images" \
  --vision_tower "${MODEL_ROOT}/clip-vit-large-patch14-336" \
  --pretrain_mm_mlp_adapter "${STAGE1}/mm_projector.bin" \
  --mm_projector_type fourier_mlp2x_gelu \
  --mm_fourier_C "${C}" \
  --mm_fourier_K_high 0 \
  --mm_fourier_spatial_keep "${S}" \
  --mm_fourier_temp_start 1.2 \
  --mm_fourier_temp_end 0.8 \
  --mm_fourier_temp_steps 4677 \
  --mm_fourier_spatial_selection_norm sparsemax \
  --mm_fourier_spatial_hidden_mult 1 \
  --mm_fourier_spatial_scale_init 1.0 \
  --mm_fourier_spatial_prior_weight_init 0.0 \
  --mm_fourier_spatial_use_norm True \
  --mm_fourier_use_polar_pe "${USE_PE}" \
  --mm_fourier_polar_pe_scale_init 0.1 \
  --mm_fourier_post_concat_norm False \
  --mm_fourier_low_idct "${LOW_IDCT}" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${STAGE2}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --seed "${SEED}" \
  --data_seed "${SEED}"

touch "${RUN_ROOT}/COMPLETE"
