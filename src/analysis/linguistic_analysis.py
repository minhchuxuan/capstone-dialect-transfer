"""
Linguistic analysis of std->dialect predictions for the report.

Produces (results/analysis_linguistic.json + .md):
  1. Region confusion matrix (marker-based): for each TARGET region, which region's
     dialect markers do the model's outputs actually contain? Reveals cross-region
     leakage (e.g. Southern outputs using Northern markers).
  2. Top learned lexical substitutions per region: standard token (in source) ->
     dialect token (in prediction) that the model introduces, ranked by frequency.
     Demonstrates the model acquired real dialect lexicon (đâu->mô, gì->chi, ...).
  3. Copy / dialectalisation summary per region.

Usage:
    python -m src.analysis.linguistic_analysis --predictions results/metrics/pred_new.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.model.config import EvalConfig

eval_cfg = EvalConfig()
REGIONS = ["northern", "central", "southern"]


def canon(r):
    return {"north": "northern", "south": "southern"}.get((r or "").lower(), (r or "").lower())


def toks(s):
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


def region_of_output(pred):
    """Marker-based: which region's markers dominate the prediction (or None)."""
    markers = eval_cfg.dialect_markers
    counts = {r: sum(1 for m in markers[r] if m in pred.lower()) for r in REGIONS}
    if sum(counts.values()) == 0:
        return None
    return max(counts, key=counts.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/metrics/pred_new.jsonl")
    ap.add_argument("--out_json", default="results/analysis_linguistic.json")
    ap.add_argument("--out_md", default="results/analysis_linguistic.md")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.predictions, encoding="utf-8")]
    std2 = [r for r in recs if r["task"].startswith("std2dialect")]

    # 1. Region confusion matrix
    confusion = {r: Counter() for r in REGIONS}
    for r in std2:
        tgt = canon(r.get("region"))
        if tgt not in REGIONS:
            continue
        out_r = region_of_output(r["prediction"])
        confusion[tgt][out_r if out_r else "none"] += 1

    # 2. Top learned substitutions per region (source standard tok -> pred dialect tok)
    #    Heuristic: for each pair, tokens the prediction INTRODUCES vs source (dialectal),
    #    associated with tokens the source DROPS (standard). Report co-occurring add/drop
    #    over single-token edits (most reliable signal).
    subs = {r: Counter() for r in REGIONS}
    for r in std2:
        tgt = canon(r.get("region"))
        if tgt not in REGIONS:
            continue
        s, p = toks(r["source"]), toks(r["prediction"])
        sset, pset = set(s), set(p)
        added = [t for t in p if t not in sset]
        dropped = [t for t in s if t not in pset]
        # Only trust clean 1-1 single substitutions
        if len(added) == 1 and len(dropped) == 1:
            subs[tgt][(dropped[0], added[0])] += 1

    # 3. Copy / dialectalisation
    summary = {}
    for r in REGIONS:
        rows = [x for x in std2 if canon(x.get("region")) == r]
        if not rows:
            continue
        copy = sum(1 for x in rows if x["prediction"].strip().lower() == x["source"].strip().lower())
        summary[r] = {"n": len(rows), "copy_rate": copy / len(rows)}

    out = {
        "region_confusion": {r: dict(confusion[r]) for r in REGIONS},
        "top_substitutions": {r: [{"std": a, "dialect": b, "count": c}
                                  for (a, b), c in subs[r].most_common(15)] for r in REGIONS},
        "summary": summary,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Markdown
    lines = ["# Linguistic analysis of std->dialect generation\n"]
    lines.append("## Region confusion matrix (rows=target, cols=marker-detected region)\n")
    lines.append("| target \\ detected | northern | central | southern | none |")
    lines.append("|---|---|---|---|---|")
    for r in REGIONS:
        c = confusion[r]
        tot = sum(c.values()) or 1
        lines.append(f"| {r} | {c.get('northern',0)} | {c.get('central',0)} | "
                     f"{c.get('southern',0)} | {c.get('none',0)} |")
    lines.append("\n## Top learned standard→dialect substitutions per region\n")
    for r in REGIONS:
        lines.append(f"\n**{r}**: " + ", ".join(
            f"{a}→{b} ({c})" for ((a, b), c) in subs[r].most_common(12)))
    lines.append("\n\n## Copy rate per region\n")
    for r, s in summary.items():
        lines.append(f"- {r}: n={s['n']}, copy_rate={s['copy_rate']:.1%}")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
