"""
Reverse the ViDia2Std corpus: swap source and target to create
standard→dialect training pairs with region-conditioned task prefixes.

Reads from data/processed/train.jsonl (dialect2std records) and writes
reverse records to the same files so the multi-task model trains on both
directions simultaneously.

Usage:
    python -m src.data.reverse_corpus
"""
import json
from pathlib import Path

from src.model.config import DataConfig

cfg = DataConfig()


def reverse_dialect2std_record(record: dict) -> dict | None:
    """Flip a dialect2std record into a std2dialect_<region> record."""
    if record["task"] != "dialect2std":
        return None
    region = record.get("region", "unknown")
    if region == "unknown":
        return None
    return {
        "task": f"std2dialect_{region}",
        "region": region,
        "source": record["target"],   # standard is now input
        "target": record["source"],   # dialect is now target
        "meta": {
            "dataset": record["meta"]["dataset"],
            "split": record["meta"]["split"],
            "direction": "reverse",
        },
    }


def process_file(path: Path) -> list[dict]:
    """Read a JSONL file, generate reverse records for dialect2std entries."""
    if not path.exists():
        print(f"  Skipping {path} (not found)")
        return []

    reverse_records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            rev = reverse_dialect2std_record(record)
            if rev:
                reverse_records.append(rev)
    return reverse_records


def main():
    print("=" * 60)
    print("Reversing ViDia2Std corpus for Standard → Dialect training")
    print("=" * 60)

    for split_name in ("train", "dev", "test"):
        path = cfg.processed_dir / f"{split_name}.jsonl"
        reverse_records = process_file(path)
        if not reverse_records:
            continue

        # Append reverse records to the existing JSONL
        with open(path, "a", encoding="utf-8") as f:
            for r in reverse_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Count by region
        region_counts: dict[str, int] = {}
        for r in reverse_records:
            region_counts[r["region"]] = region_counts.get(r["region"], 0) + 1

        print(f"  {split_name}: added {len(reverse_records)} reverse records")
        print(f"    By region: {region_counts}")


if __name__ == "__main__":
    main()
