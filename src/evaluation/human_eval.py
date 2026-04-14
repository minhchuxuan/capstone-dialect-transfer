"""
Human evaluation template generator.

Creates a structured evaluation form (CSV/TSV) for native speakers to rate
generated dialect text on:
  - Fluency (1–5)
  - Dialectal Authenticity (1–5)
  - Semantic Preservation (1–5)
  - Acceptability (binary: yes/no)

Usage:
    python -m src.evaluation.human_eval \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/human_eval_form.tsv \
        --samples_per_region 50
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.model.config import EvalConfig

eval_cfg = EvalConfig()


def sample_for_human_eval(
    pred_path: Path,
    samples_per_region: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Sample predictions for human evaluation, balanced by region."""
    random.seed(seed)

    by_region: dict[str, list] = defaultdict(list)
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Only std2dialect tasks need human eval
            if r["task"].startswith("std2dialect"):
                region = r.get("region", "unknown")
                by_region[region].append(r)

    samples = []
    for region, records in by_region.items():
        n = min(samples_per_region, len(records))
        selected = random.sample(records, n)
        for i, r in enumerate(selected):
            samples.append({
                "id": f"{region}_{i+1:03d}",
                "region": region,
                "source_standard": r["source"],
                "reference_dialect": r["target"],
                "model_output": r["prediction"],
                # Blank columns for annotators
                "fluency_1_5": "",
                "authenticity_1_5": "",
                "semantic_preservation_1_5": "",
                "acceptable_yes_no": "",
                "notes": "",
            })

    return samples


def write_eval_form(samples: list[dict], output_path: Path):
    """Write TSV evaluation form."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        print("No samples to write.")
        return

    fieldnames = list(samples[0].keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(samples)
    print(f"Human eval form: {len(samples)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate human eval form")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/human_eval_form.tsv")
    parser.add_argument("--samples_per_region", type=int,
                        default=eval_cfg.human_eval_samples_per_region)
    args = parser.parse_args()

    samples = sample_for_human_eval(
        Path(args.predictions), args.samples_per_region
    )
    write_eval_form(samples, Path(args.output))


if __name__ == "__main__":
    main()
