"""
Leave-As-Is baseline: copy input unchanged.
Serves as the lower bound for both directions.

For dialect→standard: measures how much dialect text already IS standard.
For standard→dialect: measures how much standard text already IS dialectal (should be ~0).
"""
import json
from pathlib import Path

from src.model.config import DataConfig

cfg = DataConfig()


def predict(source: str) -> str:
    """LAI simply returns the input."""
    return source


def run_on_file(input_path: Path, output_path: Path, tasks: list[str] | None = None):
    """Run LAI baseline on a JSONL file, write predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if tasks and record["task"] not in tasks:
                continue
            results.append({
                "task": record["task"],
                "region": record.get("region"),
                "source": record["source"],
                "target": record["target"],
                "prediction": predict(record["source"]),
                "baseline": "LAI",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"LAI baseline: {len(results)} predictions → {output_path}")
    return results


if __name__ == "__main__":
    test_path = cfg.processed_dir / "test.jsonl"
    run_on_file(test_path, cfg.processed_dir.parent.parent / "results" / "metrics" / "lai_predictions.jsonl")
