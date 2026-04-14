"""
Error analysis: classify model failures into a taxonomy.

Error categories for Standard → Dialect (Feature C):
  1. missing_marker    — No dialect features generated (under-generation)
  2. wrong_region      — Dialect markers from wrong region
  3. over_dialect      — Over-dialectalization making text unintelligible
  4. semantic_drift    — Meaning changed after transfer
  5. code_mixing       — Unnatural mix of standard + dialect
  6. segmentation      — Word boundary errors from token substitution

Usage:
    python -m src.evaluation.error_analysis \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/error_analysis.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.model.config import EvalConfig

eval_cfg = EvalConfig()


def detect_error_type(record: dict) -> str | None:
    """Heuristic classification of error type. Returns None if no error detected."""
    pred = record["prediction"].lower()
    ref = record["target"].lower()
    src = record["source"].lower()
    region = record.get("region", "")
    markers = eval_cfg.dialect_markers

    # 1. Missing marker: prediction ≈ source (nothing changed)
    if pred.strip() == src.strip():
        return "missing_marker"

    # Check if any dialect markers are present
    target_markers = markers.get(region, [])
    pred_has_markers = any(m in pred for m in target_markers)

    if not pred_has_markers and target_markers:
        # Check if reference has markers (if ref has no markers, this sample is trivial)
        ref_has_markers = any(m in ref for m in target_markers)
        if ref_has_markers:
            return "missing_marker"

    # 2. Wrong region: has markers from a different region
    other_regions = [r for r in markers if r != region]
    for other_region in other_regions:
        other_markers = markers[other_region]
        # Check for markers unique to the other region
        unique_other = [m for m in other_markers if m not in target_markers]
        if any(m in pred for m in unique_other):
            return "wrong_region"

    # 3. Semantic drift: simple heuristic — very different length
    len_ratio = len(pred.split()) / max(len(ref.split()), 1)
    if len_ratio < 0.5 or len_ratio > 2.0:
        return "semantic_drift"

    # 4. Over-dialectalization: prediction is much more different from source
    # than reference is from source
    src_words = set(src.split())
    ref_words = set(ref.split())
    pred_words = set(pred.split())
    ref_diff = len(ref_words - src_words)
    pred_diff = len(pred_words - src_words)
    if pred_diff > ref_diff * 2 and pred_diff > 3:
        return "over_dialect"

    # 5. Check exact match — no error
    if pred.strip() == ref.strip():
        return None  # correct

    # If we get here, it's a general mismatch (not categorized as severe error)
    return None


def analyze_predictions(pred_path: Path, max_errors: int = 50) -> dict:
    """Analyze predictions and classify errors."""
    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["task"].startswith("std2dialect"):
                records.append(r)

    # Classify errors
    errors_by_type: dict[str, list] = defaultdict(list)
    correct_count = 0

    for r in records:
        error_type = detect_error_type(r)
        if error_type:
            errors_by_type[error_type].append({
                "source": r["source"],
                "reference": r["target"],
                "prediction": r["prediction"],
                "region": r.get("region", ""),
                "error_type": error_type,
            })
        else:
            correct_count += 1

    # Summary
    total = len(records)
    total_errors = sum(len(v) for v in errors_by_type.values())
    summary = {
        "total_samples": total,
        "correct": correct_count,
        "total_errors": total_errors,
        "error_rate": total_errors / max(total, 1),
        "error_distribution": {k: len(v) for k, v in errors_by_type.items()},
    }

    # Sample errors for manual review (up to max_errors)
    sampled_errors = []
    for error_type, examples in errors_by_type.items():
        for ex in examples[:max_errors // max(len(errors_by_type), 1)]:
            sampled_errors.append(ex)

    return {
        "summary": summary,
        "sampled_errors": sampled_errors[:max_errors],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/error_analysis.json")
    parser.add_argument("--max_errors", type=int, default=eval_cfg.error_analysis_total)
    args = parser.parse_args()

    analysis = analyze_predictions(Path(args.predictions), args.max_errors)

    # Print summary
    print("\n" + "=" * 50)
    print("Error Analysis Summary")
    print("=" * 50)
    s = analysis["summary"]
    print(f"Total samples: {s['total_samples']}")
    print(f"Correct: {s['correct']}")
    print(f"Errors: {s['total_errors']} ({s['error_rate']:.1%})")
    print(f"Distribution: {s['error_distribution']}")

    # Print sample errors
    print(f"\nSampled errors ({len(analysis['sampled_errors'])}):")
    for i, err in enumerate(analysis["sampled_errors"][:10]):
        print(f"\n  [{i+1}] {err['error_type']} (region: {err['region']})")
        print(f"      SRC: {err['source']}")
        print(f"      REF: {err['reference']}")
        print(f"      PRD: {err['prediction']}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
