"""
Sampling-based back-translation augmentation for Feature C (standard -> dialect).

Method (Edunov et al., 2018 — target-real back-translation):
  - Keep the REAL dialect sentence as the TARGET (natural, fluent dialect).
  - Synthesize the standard SOURCE by running the strong dialect->standard model
    (M_fwd) with SAMPLING on the real dialect sentence, producing diverse standard
    paraphrases. Sampling (not beam) is used to maximise input-side diversity and
    avoid mode collapse (Edunov et al.'s central finding).
  - Emit (synthetic_standard -> real_dialect) pairs for task std2dialect_<region>.

We focus augmentation on the data-starved minority regions (northern, southern),
and apply light quality filtering (length ratio + the synthetic standard must
actually differ from the dialect, i.e. some normalisation happened).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.data.augment_bt_run \
        --model_path results/checkpoints/best \
        --out data/augmented/bt_round1.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.config import DataConfig

cfg = DataConfig()

# How many synthetic standard sources to sample per real dialect sentence, per region.
REGION_FANOUT = {"northern": 4, "southern": 2, "central": 0}
MAX_LEN = 128
BATCH = 64


def length_ratio_ok(a: str, b: str, tol: float = 0.5) -> bool:
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0:
        return False
    return abs(la / lb - 1.0) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="results/checkpoints/best")
    ap.add_argument("--out", default="data/augmented/bt_round1.jsonl")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()

    # Build the dialect monolingual pool from REAL dialect2std training records.
    # source=dialect (real), region=region. Fan out per region.
    pool = []  # (dialect_text, region, standard_ref)
    with open(cfg.processed_dir / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"] != "dialect2std":
                continue
            region = r.get("region")
            k = REGION_FANOUT.get(region, 0)
            for _ in range(k):
                pool.append((r["source"], region, r["target"]))

    print(f"BT pool size (dialect sentences x fanout): {len(pool)}")
    print(f"  by region: {Counter(region for _, region, _ in pool)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = Counter()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(pool), BATCH), desc="BT-sampling"):
            chunk = pool[i:i + BATCH]
            # M_fwd = dialect2std: feed the dialect sentence with the dialect2std prefix.
            inputs = [f"dialect2std: {d}" for d, _, _ in chunk]
            enc = tok(inputs, max_length=MAX_LEN, truncation=True,
                      padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, top_p=args.top_p,
                    temperature=args.temperature, num_beams=1,
                    max_new_tokens=MAX_LEN,
                )
            synth_std = tok.batch_decode(out, skip_special_tokens=True)
            for (dialect, region, std_ref), s in zip(chunk, synth_std):
                s = s.strip()
                # Quality filters: synthetic std must differ from dialect (real
                # normalisation), and have a sane length ratio.
                if not s or s.lower() == dialect.lower():
                    dropped["nochange"] += 1
                    continue
                if not length_ratio_ok(s, dialect):
                    dropped["lenratio"] += 1
                    continue
                rec = {
                    "task": f"std2dialect_{region}",
                    "region": region,
                    "source": s,             # synthetic standard (sampled)
                    "target": dialect,       # REAL dialect (kept)
                    "meta": {"dataset": "ViDia2Std-BT", "split": "train",
                             "direction": "reverse", "synthetic_source": True},
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept[region] += 1

    print(f"Kept (by region): {dict(kept)}  total={sum(kept.values())}")
    print(f"Dropped: {dict(dropped)}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
