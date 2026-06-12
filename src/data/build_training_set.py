"""
Build the final datasets for the improved multi-task model.

Produces (under data/processed/):
  - train.jsonl / dev.jsonl / test.jsonl  : CANONICAL eval sets =
        ViDia2Std (forward dialect2std + reverse std2dialect_<region>) [deduped]
        + ViLexNorm (lexnorm task, Feature B).
        These are the natural-distribution sets used for evaluation.
  - train_balanced.jsonl : TRAINING set =
        train.jsonl
        + back-translation augmentation (data/augmented/*.jsonl)
        + oversampling of the data-starved Northern std->dialect region,
        to mitigate the severe region imbalance (Central 66% / Sth 27% / Nth 7%).

Original ViDia2Std-only processed files are backed up to data/processed/orig/.

Usage:
    python -m src.data.build_training_set
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from src.model.config import DataConfig

cfg = DataConfig()
PROC = cfg.processed_dir
RAW = cfg.raw_dir
AUG = cfg.augmented_dir


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def dump_jsonl(recs: list[dict], p: Path):
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(recs):6d} -> {p.name}")


def dedupe(recs: list[dict]) -> list[dict]:
    """Remove exact (task, source, target) duplicate records, preserving order."""
    seen, out = set(), []
    for r in recs:
        k = (r["task"], r["source"].strip(), r["target"].strip())
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def load_vilexnorm_split(split: str) -> list[dict]:
    """Load one ViLexNorm CSV split as lexnorm records."""
    fp = RAW / "ViLexNorm" / f"{split}.csv"
    if not fp.exists():
        print(f"  WARNING: {fp} missing — ViLexNorm {split} skipped")
        return []
    df = pd.read_csv(fp)
    recs = []
    for _, row in df.iterrows():
        src = str(row["original"]).strip()
        tgt = str(row["normalized"]).strip()
        if not src or not tgt or src == "nan" or tgt == "nan":
            continue
        recs.append({
            "task": "lexnorm", "region": None, "source": src, "target": tgt,
            "meta": {"dataset": "ViLexNorm", "split": split, "direction": "forward"},
        })
    return recs


def main():
    # 0. Back up the original ViDia2Std-only processed files once.
    orig = PROC / "orig"
    orig.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        src = PROC / name
        dst = orig / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"backed up {name} -> orig/{name}")

    vilex_split = {"train": "train", "dev": "dev", "test": "test"}

    # 1. Build canonical sets: ViDia2Std (from backup, deduped) + ViLexNorm.
    print("\n== Canonical sets (ViDia2Std + ViLexNorm) ==")
    canonical = {}
    for split in ("train", "dev", "test"):
        vidia = load_jsonl(orig / f"{split}.jsonl")
        n_before = len(vidia)
        vidia = dedupe(vidia)
        vilex = load_vilexnorm_split(vilex_split[split])
        merged = vidia + vilex
        canonical[split] = merged
        print(f"  {split}: ViDia2Std {n_before}->{len(vidia)} (deduped) + ViLexNorm {len(vilex)} = {len(merged)}")
        print(f"         tasks: {dict(Counter(r['task'] for r in merged))}")
        dump_jsonl(merged, PROC / f"{split}.jsonl")

    # 2. Build balanced TRAIN: + back-translation + oversample Northern.
    print("\n== Balanced training set ==")
    train = list(canonical["train"])
    bt = []
    for f in sorted(AUG.glob("*.jsonl")):
        b = load_jsonl(f)
        bt.extend(b)
        print(f"  +BT {len(b)} from {f.name}")
    train_plus_bt = train + bt

    # Region counts for std2dialect after BT
    def region_counts(recs):
        return Counter(r["region"] for r in recs if r["task"].startswith("std2dialect"))
    rc = region_counts(train_plus_bt)
    print(f"  std2dialect region counts after BT: {dict(rc)}")

    # Oversample Northern std2dialect to ~ match the median region count.
    target = sorted(rc.values())[len(rc) // 2] if rc else 0  # median
    balanced = list(train_plus_bt)
    north = [r for r in train_plus_bt if r["task"] == "std2dialect_northern"]
    if north and len(north) < target:
        extra_factor = max(0, round(target / len(north)) - 1)
        for _ in range(extra_factor):
            balanced.extend(north)
        print(f"  oversampled Northern std2dialect x{extra_factor} (target~{target})")

    print(f"  final balanced train: {len(balanced)} records")
    print(f"  tasks: {dict(Counter(r['task'] for r in balanced))}")
    print(f"  std2dialect regions: {dict(region_counts(balanced))}")
    dump_jsonl(balanced, PROC / "train_balanced.jsonl")


if __name__ == "__main__":
    main()
