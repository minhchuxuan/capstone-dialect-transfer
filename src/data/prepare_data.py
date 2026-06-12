"""
Download and unify ViDia2Std, ViLexNorm, VSEC into a single JSONL schema.

Unified schema per line:
{
  "task": "dialect2std" | "std2dialect_<region>" | "lexnorm" | "spell",
  "region": "north" | "central" | "south" | null,
  "source": "<input text>",
  "target": "<output text>",
  "meta": {"dataset": "...", "split": "...", "direction": "forward"|"reverse"}
}

Usage:
    python -m src.data.prepare_data
"""
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.model.config import DataConfig

cfg = DataConfig()

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize unicode, collapse whitespace, strip URLs/mentions/hashtags."""
    if not text:
        return ""
    # NFC normalization — canonical composed form for Vietnamese diacritics
    text = unicodedata.normalize("NFC", text)
    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # Strip @mentions and #hashtags
    text = re.sub(r"[@#]\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


REGION_MAP = cfg.region_map


def normalize_region(raw_region: str) -> str:
    """Map dataset region labels to our canonical north/central/south."""
    if not raw_region:
        return "unknown"
    raw = raw_region.strip()
    return REGION_MAP.get(raw, raw.lower())


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_vidia2std() -> list[dict]:
    """Load ViDia2Std from HuggingFace and emit *forward* (dialect→std) records."""
    print("Loading ViDia2Std from HuggingFace...")
    # Load each split CSV independently to tolerate split-specific extra columns
    # (e.g., test.csv may include "sentiment" while train/dev do not).
    split_files = {
        "train": "hf://datasets/Biu3010/ViDia2Std/train.csv",
        "validation": "hf://datasets/Biu3010/ViDia2Std/dev.csv",
        "test": "hf://datasets/Biu3010/ViDia2Std/test.csv",
    }
    records = []
    for split_name, data_file in split_files.items():
        split_ds = load_dataset("csv", data_files=data_file, split="train")
        for row in tqdm(split_ds, desc=f"ViDia2Std/{split_name}"):
            dialect_text = clean_text(row.get("dialect", row.get("Dialect", "")))
            standard_text = clean_text(row.get("standard", row.get("Standard", "")))
            region_raw = row.get("region", row.get("Region", ""))
            if not dialect_text or not standard_text:
                continue
            region = normalize_region(region_raw)
            records.append({
                "task": "dialect2std",
                "region": region,
                "source": dialect_text,
                "target": standard_text,
                "meta": {
                    "dataset": "ViDia2Std",
                    "split": split_name,
                    "direction": "forward",
                },
            })
    print(f"  ViDia2Std: {len(records)} forward records")
    return records


def load_vilexnorm() -> list[dict]:
    """Load ViLexNorm from HuggingFace."""
    print("Loading ViLexNorm from HuggingFace...")
    try:
        ds = load_dataset("ngxtnhi/ViLexNorm")
    except Exception:
        # Fallback: try loading from local CSV if HF fails
        print("  HF load failed, trying local CSV...")
        csv_dir = cfg.raw_dir / "ViLexNorm"
        if not csv_dir.exists():
            print("  WARNING: ViLexNorm not found. Skipping.")
            return []
        import pandas as pd
        records = []
        for csv_file in csv_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            split_name = csv_file.stem
            for _, row in df.iterrows():
                src = clean_text(str(row.get("original", row.iloc[0])))
                tgt = clean_text(str(row.get("normalized", row.iloc[1])))
                if src and tgt:
                    records.append({
                        "task": "lexnorm",
                        "region": None,
                        "source": src,
                        "target": tgt,
                        "meta": {
                            "dataset": "ViLexNorm",
                            "split": split_name,
                            "direction": "forward",
                        },
                    })
        print(f"  ViLexNorm (local): {len(records)} records")
        return records

    records = []
    for split_name in ds:
        for row in tqdm(ds[split_name], desc=f"ViLexNorm/{split_name}"):
            src = clean_text(str(row.get("original", row.get("Original", ""))))
            tgt = clean_text(str(row.get("normalized", row.get("Normalized", ""))))
            if not src or not tgt:
                continue
            records.append({
                "task": "lexnorm",
                "region": None,
                "source": src,
                "target": tgt,
                "meta": {
                    "dataset": "ViLexNorm",
                    "split": split_name,
                    "direction": "forward",
                },
            })
    print(f"  ViLexNorm: {len(records)} records")
    return records


def load_vsec() -> list[dict]:
    """Load VSEC spelling correction dataset (optional)."""
    print("Loading VSEC...")
    vsec_dir = cfg.raw_dir / "VSEC"
    if not vsec_dir.exists() or not any(vsec_dir.iterdir()):
        print("  VSEC data not found locally. Skipping (optional dataset).")
        return []

    records = []
    for f in vsec_dir.glob("*.txt"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if "\t" in line:
                    parts = line.split("\t", 1)
                    src = clean_text(parts[0])
                    tgt = clean_text(parts[1])
                    if src and tgt and src != tgt:
                        records.append({
                            "task": "spell",
                            "region": None,
                            "source": src,
                            "target": tgt,
                            "meta": {
                                "dataset": "VSEC",
                                "split": "train",
                                "direction": "forward",
                            },
                        })
    print(f"  VSEC: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Write unified JSONL
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def split_by_dataset_split(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by their original split (train/validation/test)."""
    splits = {}
    for r in records:
        s = r["meta"]["split"]
        # Normalize split names
        if s in ("train", "training"):
            key = "train"
        elif s in ("validation", "dev", "valid"):
            key = "dev"
        elif s in ("test", "testing"):
            key = "test"
        else:
            key = "train"  # default
        splits.setdefault(key, []).append(r)
    return splits


def main():
    print("=" * 60)
    print("Preparing unified dataset")
    print("=" * 60)

    all_records = []
    all_records.extend(load_vidia2std())
    all_records.extend(load_vilexnorm())
    all_records.extend(load_vsec())

    print(f"\nTotal records: {len(all_records)}")

    # Split by original dataset splits
    by_split = split_by_dataset_split(all_records)
    for split_name, records in by_split.items():
        save_jsonl(records, cfg.processed_dir / f"{split_name}.jsonl")

    # Print summary
    print("\n--- Summary ---")
    task_counts: dict[str, int] = {}
    region_counts: dict[str, int] = {}
    for r in all_records:
        task_counts[r["task"]] = task_counts.get(r["task"], 0) + 1
        if r["region"]:
            region_counts[r["region"]] = region_counts.get(r["region"], 0) + 1

    print("By task:", task_counts)
    print("By region:", region_counts)
    print("By split:", {k: len(v) for k, v in by_split.items()})


if __name__ == "__main__":
    main()
