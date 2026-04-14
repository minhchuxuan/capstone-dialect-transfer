#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — End-to-end pipeline for Vietnamese Dialect Transfer
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh           # Full pipeline
#   ./run_pipeline.sh --step 1  # Run only step 1 (data prep)
#   ./run_pipeline.sh --step 3  # Run only step 3 (training)
# =============================================================================
set -euo pipefail

STEP="${1:---all}"
STEP_NUM="${2:-0}"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Vietnamese Bidirectional Dialect Transfer"
echo "Project dir: $PROJECT_DIR"
echo "=========================================="

# ---------- Step 1: Data Preparation ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "1" ]]; then
    echo ""
    echo ">>> Step 1: Downloading and preparing datasets..."
    python -m src.data.prepare_data
    echo ""
    echo ">>> Step 1b: Reversing ViDia2Std for Standard→Dialect..."
    python -m src.data.reverse_corpus
fi

# ---------- Step 2: Run Baselines ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "2" ]]; then
    echo ""
    echo ">>> Step 2: Running baselines..."
    python -c "
from src.baselines import lai_baseline, rule_baseline
from src.model.config import DataConfig
cfg = DataConfig()
test = cfg.processed_dir / 'test.jsonl'
out = cfg.processed_dir.parent.parent / 'results' / 'metrics'
lai_baseline.run_on_file(test, out / 'lai_predictions.jsonl')
rule_baseline.run_on_file(test, out / 'rule_predictions.jsonl')
"
fi

# ---------- Step 3: Train Multi-task Model ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "3" ]]; then
    echo ""
    echo ">>> Step 3: Training BARTpho-syllable multi-task model..."
    python -m src.model.train \
        --model_name vinai/bartpho-syllable \
        --epochs 10 \
        --batch_size 8 \
        --lr 3e-5 \
        --output_dir results/checkpoints
fi

# ---------- Step 4: Inference ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "4" ]]; then
    echo ""
    echo ">>> Step 4: Running inference on test set..."
    python -m src.model.inference \
        --model_path results/checkpoints/best \
        --input_file data/processed/test.jsonl \
        --output_file results/metrics/model_predictions.jsonl
fi

# ---------- Step 5: Evaluation ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "5" ]]; then
    echo ""
    echo ">>> Step 5: Computing evaluation metrics..."
    python -m src.evaluation.metrics \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/metrics/evaluation_results.json
fi

# ---------- Step 6: Error Analysis ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "6" ]]; then
    echo ""
    echo ">>> Step 6: Error analysis..."
    python -m src.evaluation.error_analysis \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/error_analysis.json
fi

# ---------- Step 7: Human Eval Form ----------
if [[ "$STEP" == "--all" || "$STEP_NUM" == "7" ]]; then
    echo ""
    echo ">>> Step 7: Generating human evaluation form..."
    python -m src.evaluation.human_eval \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/human_eval_form.tsv
fi

# ---------- Step 8: Demo ----------
if [[ "$STEP_NUM" == "8" ]]; then
    echo ""
    echo ">>> Step 8: Launching Gradio demo..."
    python -m demo.app --model_path results/checkpoints/best
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
