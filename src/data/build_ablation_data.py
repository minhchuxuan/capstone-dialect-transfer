"""
Build the data variants for the back-translation vs oversampling ablation.

All four cells share the SAME dialect2std + lexnorm data and differ ONLY in the
std->dialect composition, so the comparison isolates the data intervention:

  (i)   base            : reversed corpus, natural region distribution      = train.jsonl
  (ii)  +oversample     : minority regions DUPLICATED to the matched totals = train_oversample.jsonl
  (iii) +back-translate : reversed corpus + BT pairs, no oversampling       = train_bt.jsonl
  (iv)  +both           : oversample + BT (the final model)                 = train_balanced.jsonl

(ii) and (iv) reach the SAME per-region totals, so (iv) vs (ii) isolates the
value of *sampled back-translation diversity* over mere duplication.

Usage:
    python -m src.data.build_ablation_data
"""
import json
from collections import Counter
from pathlib import Path

from src.model.config import DataConfig

cfg = DataConfig()
PROC = cfg.processed_dir
AUG = cfg.augmented_dir

# Match the final (train_balanced) per-region std->dialect totals exactly.
TARGET = {"central": 7196, "southern": 8645, "northern": 7362}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def dump(recs, p):
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(recs):6d} -> {p.name}")


def replicate_to(recs, n):
    """Cycle through recs to produce exactly n records."""
    if not recs:
        return []
    out = []
    i = 0
    while len(out) < n:
        out.append(recs[i % len(recs)])
        i += 1
    return out[:n]


def main():
    base = load(PROC / "train.jsonl")
    bt = []
    for f in sorted(AUG.glob("*.jsonl")):
        bt += load(f)

    other = [r for r in base if not r["task"].startswith("std2dialect")]
    by_region = {reg: [r for r in base if r["task"] == f"std2dialect_{reg}"]
                 for reg in TARGET}

    # (ii) oversample-only: duplicate each region's REAL records up to TARGET.
    oversample = list(other)
    for reg, n in TARGET.items():
        oversample += replicate_to(by_region[reg], n)
    dump(oversample, PROC / "train_oversample.jsonl")
    print("  oversample std2dialect regions:",
          dict(Counter(r["region"] for r in oversample if r["task"].startswith("std2dialect"))))

    # (iii) BT-only: base + BT pairs, NO oversampling.
    btset = list(base) + bt
    dump(btset, PROC / "train_bt.jsonl")
    print("  bt-only std2dialect regions:",
          dict(Counter(r["region"] for r in btset if r["task"].startswith("std2dialect"))))


if __name__ == "__main__":
    main()
