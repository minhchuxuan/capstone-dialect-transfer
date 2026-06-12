#!/usr/bin/env bash
set -uo pipefail
cd /mnt/tp/minh/nlp_261/capstone-dialect-transfer
# Wait for the DDP training launcher to exit (best/ checkpoint saved on exit)
tail --pid=2335986 -f /dev/null
echo "=== training finished; starting eval $(date +%H:%M:%S) ==="
bash run_eval.sh 0
echo "=== linguistic analysis ==="
CUDA_VISIBLE_DEVICES="" /mnt/tp/miniconda/bin/python -m src.analysis.linguistic_analysis --predictions results/metrics/pred_new.jsonl
echo "=== model probe (diversity / tokenization / old-vs-new) ==="
CUDA_VISIBLE_DEVICES=0 /mnt/tp/miniconda/bin/python -m src.analysis.model_probe --new results/checkpoints_v2/best --old results/checkpoints/best
echo "=== regenerate figures with model results ==="
CUDA_VISIBLE_DEVICES="" /mnt/tp/miniconda/bin/python -m src.analysis.make_figures
echo "=== ALL POST-TRAIN DONE $(date +%H:%M:%S) ==="
