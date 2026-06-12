"""
Evaluation metrics for both directions of Vietnamese dialect transfer.

Automatic metrics:
  - BLEU (SacreBLEU)            — n-gram overlap (reported 0-100)
  - chrF / chrF++ (SacreBLEU)   — character n-gram F-score, robust for Vietnamese
  - ROUGE-L                     — longest common subsequence F1
  - METEOR (nltk)               — unigram alignment (exact/stem) F-mean
  - WER / CER                   — word/char error rate (for comparability w/ ViDia2Std)
  - BERTScore F1                — semantic similarity (multilingual model)
  - ERR (Error Reduction Rate)  — ViLexNorm normalization metric (Feature B)
  - DFR (Dialect Feature Recall)— % of reference dialect markers recalled (Feature C)
  - Dialectal Edit Recall/Prec/F1 — marker-INDEPENDENT: does the model make the
                                    token substitutions the gold reference makes?
  - Region Accuracy (marker-based) — does the output sound like the requested region?
  - Copy rate                   — fraction of outputs identical to the input (under-gen)

Usage:
    python -m src.evaluation.metrics --predictions results/metrics/model_predictions.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.model.config import EvalConfig, PROJECT_ROOT

eval_cfg = EvalConfig()


# ---------------------------------------------------------------------------
# Region handling
# ---------------------------------------------------------------------------

def canonical_region(region: str | None) -> str | None:
    """Normalize any casing/alias of a region to northern/central/southern."""
    if not region:
        return None
    r = region.strip().lower()
    return {"north": "northern", "south": "southern", "centre": "central"}.get(r, r)


def _toks(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


# ---------------------------------------------------------------------------
# BLEU / chrF
# ---------------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    from sacrebleu.metrics import BLEU
    return BLEU().corpus_score(predictions, [references]).score


def compute_chrf(predictions: list[str], references: list[str], word_order: int = 2) -> float:
    """chrF++ (word_order=2). Character n-gram F-score; robust to tokenization."""
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=word_order).corpus_score(predictions, [references]).score


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# METEOR (whitespace tokenized; exact/stem matching for Vietnamese)
# ---------------------------------------------------------------------------

def compute_meteor(predictions: list[str], references: list[str]) -> float | None:
    try:
        from nltk.translate.meteor_score import meteor_score
        vals = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
        return sum(vals) / max(len(vals), 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WER / CER (pure-python Levenshtein, corpus-level)
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compute_wer_cer(predictions: list[str], references: list[str]) -> tuple[float, float]:
    werr = wern = cerr = cern = 0
    for p, r in zip(predictions, references):
        pw, rw = p.split(), r.split()
        werr += _edit_distance(pw, rw); wern += max(len(rw), 1)
        cerr += _edit_distance(list(p), list(r)); cern += max(len(r), 1)
    return werr / max(wern, 1), cerr / max(cern, 1)


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(predictions, references, lang="vi", verbose=False)
    return F1.mean().item()


# ---------------------------------------------------------------------------
# ERR (Error Reduction Rate) — ViLexNorm (Feature B)
# ---------------------------------------------------------------------------

def compute_err(predictions: list[str], references: list[str], sources: list[str]) -> float:
    """ERR = 1 - errors_system / errors_LAI, errors = positionally-mismatched tokens."""
    def count_errors(hyps, refs):
        total = 0
        for h, r in zip(hyps, refs):
            ht, rt = h.lower().split(), r.lower().split()
            n = max(len(ht), len(rt))
            total += sum((ht[i] if i < len(ht) else "") != (rt[i] if i < len(rt) else "")
                         for i in range(n))
        return total
    errors_lai = count_errors(sources, references)
    errors_sys = count_errors(predictions, references)
    return 1.0 if errors_lai == 0 else 1.0 - errors_sys / errors_lai


# ---------------------------------------------------------------------------
# DFR (Dialect Feature Recall) — Feature C
# ---------------------------------------------------------------------------

def compute_dfr(predictions: list[str], references: list[str], regions: list[str]) -> dict:
    """% of reference dialect markers (for the target region) found in predictions."""
    markers = eval_cfg.dialect_markers
    region_scores: dict[str, list] = defaultdict(list)
    for pred, ref, region in zip(predictions, references, regions):
        region = canonical_region(region)
        if not region or region not in markers:
            continue
        rm = markers[region]
        ref_l, pred_l = ref.lower(), pred.lower()
        ref_present = [m for m in rm if m in ref_l]
        if not ref_present:
            continue
        recalled = [m for m in ref_present if m in pred_l]
        region_scores[region].append(len(recalled) / len(ref_present))
    result, all_scores = {}, []
    for region, scores in region_scores.items():
        result[f"dfr_{region}"] = sum(scores) / max(len(scores), 1)
        all_scores.extend(scores)
    result["dfr_overall"] = sum(all_scores) / max(len(all_scores), 1) if all_scores else 0.0
    return result


# ---------------------------------------------------------------------------
# Dialectal Edit Recall / Precision / F1 — marker-INDEPENDENT
# ---------------------------------------------------------------------------

def compute_edit_transfer(predictions: list[str], references: list[str], sources: list[str]) -> dict:
    """Among the tokens the GOLD reference introduces vs the standard source
    (the dialectal substitutions), how many does the model also introduce?
    Recall = |gold_add & model_add| / |gold_add|, Precision likewise over model_add.
    """
    rsum = psum = n = 0
    for pred, ref, src in zip(predictions, references, sources):
        s, t, p = set(_toks(src)), set(_toks(ref)), set(_toks(pred))
        gold_add = t - s
        if not gold_add:
            continue
        model_add = p - s
        inter = gold_add & model_add
        rsum += len(inter) / len(gold_add)
        psum += (len(inter) / len(model_add)) if model_add else 0.0
        n += 1
    rec = rsum / max(n, 1)
    prec = psum / max(n, 1)
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
    return {"edit_recall": rec, "edit_precision": prec, "edit_f1": f1, "edit_n": n}


# ---------------------------------------------------------------------------
# Region accuracy (marker-based classifier proxy)
# ---------------------------------------------------------------------------

def region_accuracy_marker(predictions: list[str], regions: list[str]) -> dict:
    """Classify each prediction to the region whose markers it contains most of;
    accuracy = fraction matching the requested region (over predictions that
    contain at least one marker of any region). Also report confusion-ish coverage.
    """
    markers = eval_cfg.dialect_markers
    correct = total = no_marker = 0
    for pred, region in zip(predictions, regions):
        region = canonical_region(region)
        if region not in markers:
            continue
        pl = pred.lower()
        counts = {r: sum(1 for m in ms if m in pl) for r, ms in markers.items()}
        if sum(counts.values()) == 0:
            no_marker += 1
            total += 1
            continue
        pred_region = max(counts, key=counts.get)
        correct += int(pred_region == region)
        total += 1
    return {
        "region_acc": correct / max(total, 1),
        "region_n": total,
        "region_no_marker_frac": no_marker / max(total, 1),
    }


def copy_rate(predictions: list[str], sources: list[str]) -> float:
    return sum(p.strip().lower() == s.strip().lower()
               for p, s in zip(predictions, sources)) / max(len(predictions), 1)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_predictions(pred_path: Path, with_bertscore: bool = True) -> dict:
    records = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    by_task: dict[str, list] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    all_results = {}
    for task, trs in by_task.items():
        preds = [r["prediction"] for r in trs]
        refs = [r["target"] for r in trs]
        srcs = [r["source"] for r in trs]
        regions = [r.get("region", "") for r in trs]

        m = {
            "count": len(trs),
            "bleu": compute_bleu(preds, refs),
            "chrf": compute_chrf(preds, refs),
            "rouge_l": compute_rouge_l(preds, refs),
            "copy_rate": copy_rate(preds, srcs),
        }
        meteor = compute_meteor(preds, refs)
        if meteor is not None:
            m["meteor"] = meteor

        if task == "dialect2std":
            wer, cer = compute_wer_cer(preds, refs)
            m["wer"], m["cer"] = wer, cer

        if with_bertscore:
            try:
                m["bertscore_f1"] = compute_bertscore(preds, refs)
            except Exception as e:
                m["bertscore_f1"] = f"error: {e}"

        if task == "lexnorm":
            m["err"] = compute_err(preds, refs, srcs)

        if task.startswith("std2dialect"):
            m.update(compute_dfr(preds, refs, regions))
            m.update(compute_edit_transfer(preds, refs, srcs))
            m.update(region_accuracy_marker(preds, regions))

        all_results[task] = m
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_bertscore", action="store_true")
    args = parser.parse_args()

    results = evaluate_predictions(Path(args.predictions), with_bertscore=not args.no_bertscore)
    for task, metrics in results.items():
        print(f"\n{'='*52}\nTask: {task}\n{'='*52}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = Path(args.output) if args.output else Path(args.predictions).parent / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
