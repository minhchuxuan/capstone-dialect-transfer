# Vietnamese Bidirectional Dialect Transfer

**Capstone Project — IT4772E Natural Language Processing, HUST 2025-2**

Bidirectional transfer between standard Vietnamese and regional dialects (Northern, Central, Southern) using multi-task seq2seq models with iterative back-translation.

## Features

- **Feature A — Dialect → Standard:** Normalize regional dialect text to standard Vietnamese (ViDia2Std)
- **Feature B — Lexical Normalization:** Convert teencode/social media text to standard Vietnamese (ViLexNorm)
- **Feature C — Standard → Dialect:** Generate regional dialect text from standard Vietnamese (novel contribution)

## Results (improved multi-task model, test set)

| Feature | Task | Headline metric | Context |
|---|---|---|---|
| A | dialect→standard | BLEU **82.10**, WER **0.112**, CER **0.069** | beats ViDia2Std's best mBART-large-50 (0.123/0.075), 396M vs 611M params |
| B | lexical normalization | ERR **0.593** | exceeds ViLexNorm BARTpho-syllable (0.5774) |
| C | std→dialect (central) | DFR 0.901, Edit-Recall 0.714 | region-conditioned generation (novel) |
| C | std→dialect (southern) | DFR 0.897, Edit-Recall 0.646 | |
| C | std→dialect (northern) | DFR 0.897, Edit-Recall **0.635** (from 0.385) | copy-rate 0.209→0.090 via rebalancing + back-translation |

Full report: [`report/main.pdf`](report/main.pdf) (17 pp); slides: [`report/slides.pdf`](report/slides.pdf).

## Quick Start (reproduce the improved pipeline)

Uses the conda **base** env (has torch/transformers/sacrebleu/bert-score/gradio). Two GPUs assumed.

```bash
PY=/mnt/tp/miniconda/bin/python

# 1. Data: ViDia2Std is pulled from HF; ViLexNorm CSVs live in data/raw/ViLexNorm/.
#    (Original ViDia2Std-only processed files are backed up to data/processed/orig/.)
$PY -m src.data.prepare_data && $PY -m src.data.reverse_corpus      # base corpus (already done)
CUDA_VISIBLE_DEVICES=0 $PY -m src.data.augment_bt_run \             # sampling back-translation
    --model_path results/checkpoints/best --out data/augmented/bt_round1.jsonl
$PY -m src.data.build_training_set                                  # +ViLexNorm, dedupe, oversample → train_balanced.jsonl

# 2. Train improved multi-task model (DDP, both GPUs, ~25 min)
CUDA_VISIBLE_DEVICES=0,1 /mnt/tp/miniconda/bin/torchrun --nproc_per_node=2 -m src.model.train \
    --model_name vinai/bartpho-syllable --epochs 10 --batch_size 32 --grad_accum 1 --lr 5e-5 \
    --train_file train_balanced.jsonl --dev_file dev_eval.jsonl --output_dir results/checkpoints_v2

# 3. Evaluate (old vs improved) + baselines + analysis + figures
bash run_eval.sh 0                                                  # inference + metrics for both models
$PY -m src.analysis.linguistic_analysis --predictions results/metrics/pred_new.jsonl
CUDA_VISIBLE_DEVICES=0 $PY -m src.analysis.model_probe              # diversity / tokenization / old-vs-new
$PY -m src.analysis.make_figures                                    # 6 report figures → results/figures/

# 4. Report + slides (XeTeX engine; renders Vietnamese via fontspec)
cd report && /mnt/tp/miniconda/envs/mpg/bin/tectonic -r 10 main.tex && \
             /mnt/tp/miniconda/envs/mpg/bin/tectonic -r 10 slides.tex

# Demo
$PY -m demo.app --model_path results/checkpoints_v2/best
```

> **Model note:** the trained model is `bartpho-syllable` (the **large**, 396M variant — the mBART-architecture flagship), not the 132M base. The implementation plan's "132M" conflated the base/large checkpoints; the large model is the right, stronger choice and matches the ViDia2Std SOTA on Feature A.

## Project Structure

```
capstone-dialect-transfer/
├── data/                       # Datasets (raw, processed, augmented)
├── src/
│   ├── data/                   # Data pipeline (download, reverse, back-translate)
│   ├── baselines/              # LAI, dictionary, rule-based, retrieval
│   ├── model/                  # Config, training, inference
│   ├── evaluation/             # Metrics, human eval, error analysis
│   └── analysis/               # VnCoreNLP analysis, dialect classifier
├── demo/                       # Gradio web app
├── notebooks/                  # Exploration and analysis notebooks
├── results/                    # Metrics, figures, checkpoints
├── requirements.txt
└── run_pipeline.sh             # End-to-end pipeline script
```

## Datasets

| Dataset | Task | Size |
|---------|------|------|
| [ViDia2Std](https://huggingface.co/datasets/Biu3010/ViDia2Std) | Dialect → Standard | 13,657 pairs |
| [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) | Lexical Normalization | 10,467 pairs |
| [VSEC](https://github.com/VSEC2021/VSEC) | Spelling Correction (optional) | 9,341 sentences |

## Models

| Model | Role |
|-------|------|
| [BARTpho-syllable](https://huggingface.co/vinai/bartpho-syllable) (132M) | Primary seq2seq model |
| [ViT5-base](https://huggingface.co/VietAI/vit5-base) (310M) | Comparison model |
| [PhoBERT-base](https://huggingface.co/vinai/phobert-base) | Dialect region classifier |
