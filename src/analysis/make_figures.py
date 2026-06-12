"""
Generate publication figures for the report into results/figures/ (PDF + PNG).

Reads:
  - data/processed/*.jsonl                 (data composition / imbalance)
  - results/metrics/baselines_summary.json (baseline metrics)
  - results/metrics/eval_old.json          (old model, if present)
  - results/metrics/eval_new.json          (improved model, if present)

Figures that need the model eval files are skipped gracefully if absent
(re-run this script after evaluation to fill them in).

Usage:
    python -m src.analysis.make_figures
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model.config import DataConfig

cfg = DataConfig()
FIG = cfg.processed_dir.parent.parent / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MET = cfg.processed_dir.parent.parent / "results" / "metrics"

# Okabe-Ito colorblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "black": "#000000",
      "grey": "#999999"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "legend.frameon": False,
})

REGIONS = ["northern", "central", "southern"]
RLAB = {"northern": "Northern", "central": "Central", "southern": "Southern"}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if Path(p).exists() else []


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def fig_region_imbalance():
    """std2dialect region counts: canonical train vs balanced train."""
    canon = load_jsonl(cfg.processed_dir / "train.jsonl")
    bal = load_jsonl(cfg.processed_dir / "train_balanced.jsonl")
    c_canon = Counter(r["region"] for r in canon if r["task"].startswith("std2dialect"))
    c_bal = Counter(r["region"] for r in bal if r["task"].startswith("std2dialect"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(REGIONS)); w = 0.38
    ax.bar([i - w / 2 for i in x], [c_canon[r] for r in REGIONS], w,
           label="Original (reverse corpus)", color=OI["grey"])
    ax.bar([i + w / 2 for i in x], [c_bal[r] for r in REGIONS], w,
           label="+ Back-translation + oversampling", color=OI["blue"])
    for i, r in enumerate(REGIONS):
        ax.text(i - w / 2, c_canon[r] + 80, str(c_canon[r]), ha="center", fontsize=8)
        ax.text(i + w / 2, c_bal[r] + 80, str(c_bal[r]), ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([RLAB[r] for r in REGIONS])
    ax.set_ylabel("Training pairs (std$\\rightarrow$dialect)")
    ax.set_title("Region imbalance and rebalancing")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "region_imbalance")


def fig_dataset_composition():
    counts = {}
    for sp in ["train", "dev", "test"]:
        recs = load_jsonl(cfg.processed_dir / f"{sp}.jsonl")
        counts[sp] = Counter(r["task"] for r in recs)
    tasks = ["dialect2std", "std2dialect_central", "std2dialect_southern",
             "std2dialect_northern", "lexnorm"]
    tlab = ["A: dialect→std", "C: std→central", "C: std→southern",
            "C: std→northern", "B: lexnorm"]
    colors = [OI["blue"], OI["green"], OI["sky"], OI["orange"], OI["purple"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    splits = ["train", "dev", "test"]
    bottoms = [0, 0, 0]
    for t, lab, col in zip(tasks, tlab, colors):
        vals = [counts[s].get(t, 0) for s in splits]
        ax.bar(splits, vals, bottom=bottoms, label=lab, color=col)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, s in enumerate(splits):
        ax.text(i, bottoms[i] + 200, str(bottoms[i]), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of examples")
    ax.set_title("Dataset composition by task and split")
    ax.legend(loc="upper right", fontsize=8, ncol=1)
    save(fig, "dataset_composition")


def fig_copy_vs_model():
    """Copy (LAI) BLEU vs model BLEU per std2dialect region — shows northern≈standard."""
    bs = json.load(open(MET / "baselines_summary.json")) if (MET / "baselines_summary.json").exists() else {}
    lai = bs.get("LAI (leave-as-is)", {})
    new = json.load(open(MET / "eval_new.json")) if (MET / "eval_new.json").exists() else {}
    old = json.load(open(MET / "eval_old.json")) if (MET / "eval_old.json").exists() else {}
    tasks = [f"std2dialect_{r}" for r in REGIONS]
    copy_b = [lai.get(t, {}).get("bleu", 0) for t in tasks]
    old_b = [old.get(t, {}).get("bleu", 0) for t in tasks]
    new_b = [new.get(t, {}).get("bleu", 0) for t in tasks]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = range(len(REGIONS)); w = 0.26
    ax.bar([i - w for i in x], copy_b, w, label="Copy (LAI)", color=OI["grey"])
    if any(old_b):
        ax.bar([i for i in x], old_b, w, label="Model (baseline)", color=OI["orange"])
    if any(new_b):
        ax.bar([i + w for i in x], new_b, w, label="Model (improved)", color=OI["blue"])
    ax.set_xticks(list(x)); ax.set_xticklabels([RLAB[r] for r in REGIONS])
    ax.set_ylabel("BLEU"); ax.set_title("Standard→Dialect: copy baseline vs model")
    ax.legend(fontsize=9)
    save(fig, "copy_vs_model_bleu")


def fig_baseline_comparison():
    bs = json.load(open(MET / "baselines_summary.json")) if (MET / "baselines_summary.json").exists() else {}
    if not bs:
        print("  (skip baseline_comparison: no summary)"); return
    tasks = ["dialect2std", "std2dialect_central", "std2dialect_southern",
             "std2dialect_northern", "lexnorm"]
    tlab = ["A:d→s", "C:central", "C:southern", "C:northern", "B:lexnorm"]
    names = list(bs.keys())
    colors = [OI["grey"], OI["orange"], OI["green"], OI["sky"], OI["blue"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(tasks)); n = len(names); w = 0.8 / n
    for j, nm in enumerate(names):
        vals = [bs[nm].get(t, {}).get("bleu", 0) for t in tasks]
        ax.bar([i + (j - n / 2) * w + w / 2 for i in x], vals, w, label=nm, color=colors[j % len(colors)])
    ax.set_xticks(list(x)); ax.set_xticklabels(tlab)
    ax.set_ylabel("BLEU"); ax.set_title("Baseline BLEU by task")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "baseline_comparison")


def fig_dfr_by_region():
    new = json.load(open(MET / "eval_new.json")) if (MET / "eval_new.json").exists() else {}
    old = json.load(open(MET / "eval_old.json")) if (MET / "eval_old.json").exists() else {}
    if not (new or old):
        print("  (skip dfr_by_region: no model eval yet)"); return
    tasks = [f"std2dialect_{r}" for r in REGIONS]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = range(len(REGIONS)); w = 0.35
    if old:
        ax.bar([i - w / 2 for i in x], [old.get(t, {}).get("dfr_overall", 0) for t in tasks], w,
               label="Baseline model", color=OI["orange"])
    if new:
        ax.bar([i + w / 2 for i in x], [new.get(t, {}).get("dfr_overall", 0) for t in tasks], w,
               label="Improved model", color=OI["blue"])
    ax.set_xticks(list(x)); ax.set_xticklabels([RLAB[r] for r in REGIONS])
    ax.set_ylabel("Dialect Feature Recall"); ax.set_ylim(0, 1)
    ax.set_title("DFR by region")
    ax.legend(fontsize=9)
    save(fig, "dfr_by_region")


def fig_main_results():
    new = json.load(open(MET / "eval_new.json")) if (MET / "eval_new.json").exists() else {}
    old = json.load(open(MET / "eval_old.json")) if (MET / "eval_old.json").exists() else {}
    if not (new and old):
        print("  (skip main_results: need both eval_old and eval_new)"); return
    tasks = ["dialect2std", "std2dialect_central", "std2dialect_southern",
             "std2dialect_northern", "lexnorm"]
    tlab = ["A:d→s", "C:central", "C:southern", "C:northern", "B:lexnorm"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = range(len(tasks)); w = 0.38
    ax.bar([i - w / 2 for i in x], [old.get(t, {}).get("bleu", 0) for t in tasks], w,
           label="Baseline model", color=OI["orange"])
    ax.bar([i + w / 2 for i in x], [new.get(t, {}).get("bleu", 0) for t in tasks], w,
           label="Improved model", color=OI["blue"])
    ax.set_xticks(list(x)); ax.set_xticklabels(tlab)
    ax.set_ylabel("BLEU"); ax.set_title("Improved vs baseline model (test BLEU)")
    ax.legend(fontsize=9)
    save(fig, "main_results")


if __name__ == "__main__":
    print(f"Writing figures to {FIG}")
    fig_region_imbalance()
    fig_dataset_composition()
    fig_copy_vs_model()
    fig_baseline_comparison()
    fig_dfr_by_region()
    fig_main_results()
    print("done")
