#!/usr/bin/env bash
# Post-training evaluation: run inference for OLD (baseline) and NEW (improved)
# models on the canonical test set, then compute the full metric suite for both.
set -euo pipefail
cd "$(dirname "$0")"
PY=/mnt/tp/miniconda/bin/python
GPU="${1:-0}"   # which GPU to use for inference

OLD=results/checkpoints/best
NEW=results/checkpoints_v2/best
TEST=data/processed/test.jsonl

echo ">>> Inference: improved model ($NEW) on full test set (all tasks)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.model.inference \
    --model_path "$NEW" --input_file "$TEST" \
    --output_file results/metrics/pred_new.jsonl --batch_size 32

echo ">>> Inference: baseline model ($OLD) on test set (dialect tasks only; it has no lexnorm)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.model.inference \
    --model_path "$OLD" --input_file "$TEST" \
    --output_file results/metrics/pred_old.jsonl --batch_size 32 \
    --tasks dialect2std std2dialect_central std2dialect_southern std2dialect_northern

echo ">>> Evaluate improved model (with BERTScore)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.evaluation.metrics \
    --predictions results/metrics/pred_new.jsonl --output results/metrics/eval_new.json

echo ">>> Evaluate baseline model (with BERTScore)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.evaluation.metrics \
    --predictions results/metrics/pred_old.jsonl --output results/metrics/eval_old.json

echo ">>> Error taxonomy on improved model"
CUDA_VISIBLE_DEVICES=$GPU $PY -m src.evaluation.error_analysis \
    --predictions results/metrics/pred_new.jsonl --output results/error_analysis_new.json || true

echo ">>> Done. eval_old.json / eval_new.json written."
