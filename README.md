# Vietnamese Bidirectional Dialect Transfer

**Capstone Project — IT4772E Natural Language Processing, HUST 2025-2**

Bidirectional transfer between standard Vietnamese and regional dialects (Northern, Central, Southern) using multi-task seq2seq models with iterative back-translation.

## Features

- **Feature A — Dialect → Standard:** Normalize regional dialect text to standard Vietnamese (ViDia2Std)
- **Feature B — Lexical Normalization:** Convert teencode/social media text to standard Vietnamese (ViLexNorm)
- **Feature C — Standard → Dialect:** Generate regional dialect text from standard Vietnamese (novel contribution)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Download and prepare datasets
python -m src.data.prepare_data
python -m src.data.reverse_corpus

# Step 2: Run baselines
python -c "
from src.baselines import lai_baseline, rule_baseline
from src.model.config import DataConfig
cfg = DataConfig()
test = cfg.processed_dir / 'test.jsonl'
out = cfg.processed_dir.parent.parent / 'results' / 'metrics'
lai_baseline.run_on_file(test, out / 'lai_predictions.jsonl')
rule_baseline.run_on_file(test, out / 'rule_predictions.jsonl')
"

# Step 3: Train model
python -m src.model.train --model_name vinai/bartpho-syllable --epochs 10

# Step 4: Inference
python -m src.model.inference --model_path results/checkpoints/best

# Step 5: Evaluate
python -m src.evaluation.metrics --predictions results/metrics/model_predictions.jsonl

# Step 6: Launch demo
python -m demo.app --model_path results/checkpoints/best
# Or without model (rule-based only):
python -m demo.app --use_rules_only
```

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
