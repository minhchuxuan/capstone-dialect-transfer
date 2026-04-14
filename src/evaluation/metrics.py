"""
Evaluation metrics for both directions of dialect transfer.

Metrics:
  - BLEU (SacreBLEU) — n-gram overlap
  - ROUGE-L — longest common subsequence
  - BERTScore — semantic similarity
  - ERR (Error Reduction Rate) — for lexical normalization (ViLexNorm standard)
  - DFR (Dialect Feature Recall) — custom metric for standard→dialect

Usage:
    python -m src.evaluation.metrics --predictions results/metrics/model_predictions.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.model.config import EvalConfig, PROJECT_ROOT

eval_cfg = EvalConfig()


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus-level BLEU using SacreBLEU."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    result = bleu.corpus_score(predictions, [references])
    return result.score


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Average ROUGE-L F1."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        scores.append(s["rougeL"].fmeasure)
    return sum(scores) / max(len(scores), 1)


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """Average BERTScore F1 (using multilingual model for Vietnamese)."""
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="vi",  # Vietnamese
        verbose=False,
    )
    return F1.mean().item()


# ---------------------------------------------------------------------------
# ERR (Error Reduction Rate) — ViLexNorm standard
# ---------------------------------------------------------------------------

def compute_err(
    predictions: list[str],
    references: list[str],
    sources: list[str],
) -> float:
    """Error Reduction Rate with LAI baseline.

    ERR = 1 - (errors_system / errors_LAI)
    where errors = number of tokens that differ from reference.
    """
    def count_errors(hyps, refs):
        total = 0
        for h, r in zip(hyps, refs):
            h_toks = h.lower().split()
            r_toks = r.lower().split()
            max_len = max(len(h_toks), len(r_toks))
            errors = 0
            for i in range(max_len):
                h_t = h_toks[i] if i < len(h_toks) else ""
                r_t = r_toks[i] if i < len(r_toks) else ""
                if h_t != r_t:
                    errors += 1
            total += errors
        return total

    errors_lai = count_errors(sources, references)  # LAI = copy input
    errors_sys = count_errors(predictions, references)

    if errors_lai == 0:
        return 1.0  # perfect LAI means task is trivial
    return 1.0 - (errors_sys / errors_lai)


# ---------------------------------------------------------------------------
# DFR (Dialect Feature Recall) — Custom metric for std→dialect
# ---------------------------------------------------------------------------

def compute_dfr(
    predictions: list[str],
    references: list[str],
    regions: list[str],
) -> dict[str, float]:
    """Dialect Feature Recall: % of reference dialect markers found in predictions.

    Returns per-region DFR and overall DFR.
    """
    markers = eval_cfg.dialect_markers
    region_scores: dict[str, list] = defaultdict(list)

    for pred, ref, region in zip(predictions, references, regions):
        if not region or region not in markers:
            continue
        region_markers = markers[region]
        ref_lower = ref.lower()
        pred_lower = pred.lower()

        # Which markers are in the reference?
        ref_present = [m for m in region_markers if m in ref_lower]
        if not ref_present:
            continue  # skip if reference has no markers

        # Which of those are in the prediction?
        recalled = [m for m in ref_present if m in pred_lower]
        score = len(recalled) / len(ref_present)
        region_scores[region].append(score)

    result = {}
    all_scores = []
    for region, scores in region_scores.items():
        avg = sum(scores) / max(len(scores), 1)
        result[f"dfr_{region}"] = avg
        all_scores.extend(scores)

    result["dfr_overall"] = sum(all_scores) / max(len(all_scores), 1)
    return result


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_predictions(pred_path: Path) -> dict:
    """Load predictions JSONL, compute all metrics grouped by task."""
    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Group by task
    by_task: dict[str, list] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    all_results = {}

    for task, task_records in by_task.items():
        preds = [r["prediction"] for r in task_records]
        refs = [r["target"] for r in task_records]
        srcs = [r["source"] for r in task_records]
        regions = [r.get("region", "") for r in task_records]

        metrics = {
            "count": len(task_records),
            "bleu": compute_bleu(preds, refs),
            "rouge_l": compute_rouge_l(preds, refs),
        }

        # BERTScore — can be slow, compute on demand
        try:
            metrics["bertscore_f1"] = compute_bertscore(preds, refs)
        except Exception as e:
            metrics["bertscore_f1"] = f"error: {e}"

        # ERR for lexnorm tasks
        if task == "lexnorm":
            metrics["err"] = compute_err(preds, refs, srcs)

        # DFR for std2dialect tasks
        if task.startswith("std2dialect"):
            dfr = compute_dfr(preds, refs, regions)
            metrics.update(dfr)

        all_results[task] = metrics

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSONL file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    results = evaluate_predictions(Path(args.predictions))

    # Print results
    for task, metrics in results.items():
        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.predictions).parent / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
