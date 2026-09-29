#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${1:?benchmark key required}"
: "${MODEL_PATH:?}"
: "${RESULT_ROOT:?}"
: "${DATA_ROOT:?}"

mkdir -p "${RESULT_ROOT}/${BENCHMARK}"
OUT="${RESULT_ROOT}/${BENCHMARK}/answers.jsonl"
EVAL="${DATA_ROOT}/eval"

case "${BENCHMARK}" in
  gqa)
    python -m llava.eval.model_vqa_loader \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/gqa/llava_gqa_testdev_balanced.jsonl" \
      --image-folder "${EVAL}/gqa/data/images" \
      --answers-file "${OUT}" \
      --temperature 0 --conv-mode vicuna_v1
    python scripts/convert_gqa_for_eval.py \
      --src "${OUT}" \
      --dst "${EVAL}/gqa/data/testdev_balanced_predictions.json"
    cp "${EVAL}/gqa/data/testdev_balanced_predictions.json" \
      "${RESULT_ROOT}/${BENCHMARK}/testdev_balanced_predictions.json"
    (
      cd "${EVAL}/gqa/data"
      python eval/eval.py \
        --tier testdev_balanced \
        > "${RESULT_ROOT}/${BENCHMARK}/score.txt"
    )
    ;;
  mmbench_en|mmbench_cn)
    if [[ "${BENCHMARK}" == mmbench_en ]]; then
      SPLIT=mmbench_dev_20230712
      LANG=()
    else
      SPLIT=mmbench_dev_cn_20231003
      LANG=(--lang cn)
    fi
    python -m llava.eval.model_vqa_mmbench \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/mmbench/${SPLIT}.tsv" \
      --answers-file "${OUT}" \
      "${LANG[@]}" \
      --single-pred-prompt --temperature 0 --conv-mode vicuna_v1
    mkdir -p "${RESULT_ROOT}/${BENCHMARK}/upload"
    python scripts/convert_mmbench_for_submission.py \
      --annotation-file "${EVAL}/mmbench/${SPLIT}.tsv" \
      --result-dir "${RESULT_ROOT}/${BENCHMARK}" \
      --upload-dir "${RESULT_ROOT}/${BENCHMARK}/upload" \
      --experiment answers
    ;;
  mme)
    MME_EXPERIMENT="${METHOD}_k${BUDGET}_seed${SEED}"
    python -m llava.eval.model_vqa_loader \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/mme/llava_mme.jsonl" \
      --image-folder "${EVAL}/mme/MME_Benchmark_release_version" \
      --answers-file "${OUT}" \
      --temperature 0 --conv-mode vicuna_v1
    mkdir -p "${EVAL}/mme/answers"
    cp "${OUT}" "${EVAL}/mme/answers/${MME_EXPERIMENT}.jsonl"
    (
      cd "${EVAL}/mme"
      python convert_answer_to_mme.py \
        --experiment "${MME_EXPERIMENT}"
      cd eval_tool
      python calculation.py \
        --results_dir "answers/${MME_EXPERIMENT}" \
        > "${RESULT_ROOT}/${BENCHMARK}/score.txt"
    )
    mkdir -p "${RESULT_ROOT}/${BENCHMARK}/official_answers"
    cp -a "${EVAL}/mme/eval_tool/answers/${MME_EXPERIMENT}/." \
      "${RESULT_ROOT}/${BENCHMARK}/official_answers/"
    ;;
  pope)
    python -m llava.eval.model_vqa_loader \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
      --image-folder "${EVAL}/pope/coco-val2014" \
      --answers-file "${OUT}" \
      --temperature 0 --conv-mode vicuna_v1
    python llava/eval/eval_pope.py \
      --annotation-dir "${EVAL}/pope/coco" \
      --question-file "${EVAL}/pope/llava_pope_test.jsonl" \
      --result-file "${OUT}" \
      > "${RESULT_ROOT}/${BENCHMARK}/score.txt"
    ;;
  scienceqa)
    python -m llava.eval.model_vqa_science \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/scienceqa/llava_test_CQM-A.json" \
      --image-folder "${EVAL}/scienceqa/images/test" \
      --answers-file "${OUT}" \
      --single-pred-prompt --temperature 0 --conv-mode vicuna_v1
    python llava/eval/eval_science_qa.py \
      --base-dir "${EVAL}/scienceqa" \
      --result-file "${OUT}" \
      --output-file "${RESULT_ROOT}/${BENCHMARK}/output.jsonl" \
      --output-result "${RESULT_ROOT}/${BENCHMARK}/score.json"
    ;;
  textvqa)
    python -m llava.eval.model_vqa_loader \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
      --image-folder "${EVAL}/textvqa/train_images" \
      --answers-file "${OUT}" \
      --temperature 0 --conv-mode vicuna_v1
    python -m llava.eval.eval_textvqa \
      --annotation-file "${EVAL}/textvqa/TextVQA_0.5.1_val.json" \
      --result-file "${OUT}" \
      > "${RESULT_ROOT}/${BENCHMARK}/score.txt"
    ;;
  mmvet)
    python -m llava.eval.model_vqa \
      --model-path "${MODEL_PATH}" \
      --question-file "${EVAL}/mmvet/llava-mm-vet.jsonl" \
      --image-folder "${EVAL}/mmvet/images" \
      --answers-file "${OUT}" \
      --temperature 0 --conv-mode vicuna_v1
    python scripts/convert_mmvet_for_eval.py \
      --src "${OUT}" \
      --dst "${RESULT_ROOT}/${BENCHMARK}/answers.json"
    ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}" >&2
    exit 2
    ;;
esac
