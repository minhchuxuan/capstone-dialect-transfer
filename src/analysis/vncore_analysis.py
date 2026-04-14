"""
VnCoreNLP-based linguistic analysis.

Compares POS tag and dependency parse distributions before/after
dialect transfer to demonstrate NLP fundamentals understanding.

Note: VnCoreNLP requires Java. Install via:
    pip install py_vncorenlp
    python -c "import py_vncorenlp; py_vncorenlp.download_model(save_dir='./vncorenlp')"

Usage:
    python -m src.analysis.vncore_analysis \
        --predictions results/metrics/model_predictions.jsonl \
        --output results/vncore_analysis.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_vncorenlp(model_dir: str = "./vncorenlp"):
    """Load VnCoreNLP model. Returns None if not available."""
    try:
        import py_vncorenlp
        model = py_vncorenlp.VnCoreNLP(save_dir=model_dir)
        return model
    except Exception as e:
        print(f"VnCoreNLP not available: {e}")
        print("Install: pip install py_vncorenlp")
        print("Download: python -c \"import py_vncorenlp; py_vncorenlp.download_model(save_dir='./vncorenlp')\"")
        return None


def analyze_pos(vncore, text: str) -> Counter:
    """Get POS tag distribution for a text."""
    pos_counts = Counter()
    try:
        result = vncore.annotate_text(text)
        for sent in result:
            for token in sent:
                pos_counts[token["posTag"]] += 1
    except Exception:
        pass
    return pos_counts


def analyze_deps(vncore, text: str) -> Counter:
    """Get dependency relation distribution."""
    dep_counts = Counter()
    try:
        result = vncore.annotate_text(text)
        for sent in result:
            for token in sent:
                dep_counts[token["depLabel"]] += 1
    except Exception:
        pass
    return dep_counts


def compare_distributions(
    pred_path: Path,
    model_dir: str = "./vncorenlp",
    max_samples: int = 200,
) -> dict:
    """Compare POS/dependency distributions: source vs prediction vs reference."""
    vncore = load_vncorenlp(model_dir)
    if vncore is None:
        return {"error": "VnCoreNLP not available"}

    records = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # Sample to keep runtime manageable
    if len(records) > max_samples:
        import random
        random.seed(42)
        records = random.sample(records, max_samples)

    # Aggregate POS distributions
    pos_source = Counter()
    pos_pred = Counter()
    pos_ref = Counter()
    dep_source = Counter()
    dep_pred = Counter()
    dep_ref = Counter()

    for r in records:
        pos_source += analyze_pos(vncore, r["source"])
        pos_pred += analyze_pos(vncore, r["prediction"])
        pos_ref += analyze_pos(vncore, r["target"])
        dep_source += analyze_deps(vncore, r["source"])
        dep_pred += analyze_deps(vncore, r["prediction"])
        dep_ref += analyze_deps(vncore, r["target"])

    return {
        "samples_analyzed": len(records),
        "pos_distributions": {
            "source": dict(pos_source.most_common()),
            "prediction": dict(pos_pred.most_common()),
            "reference": dict(pos_ref.most_common()),
        },
        "dep_distributions": {
            "source": dict(dep_source.most_common()),
            "prediction": dict(dep_pred.most_common()),
            "reference": dict(dep_ref.most_common()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="VnCoreNLP analysis")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/vncore_analysis.json")
    parser.add_argument("--model_dir", type=str, default="./vncorenlp")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    analysis = compare_distributions(
        Path(args.predictions), args.model_dir, args.max_samples,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"VnCoreNLP analysis saved to {out_path}")

    # Print summary
    if "error" not in analysis:
        print(f"\nSamples analyzed: {analysis['samples_analyzed']}")
        print("\nTop POS tags (source → prediction → reference):")
        for tag in list(analysis["pos_distributions"]["source"].keys())[:10]:
            s = analysis["pos_distributions"]["source"].get(tag, 0)
            p = analysis["pos_distributions"]["prediction"].get(tag, 0)
            r = analysis["pos_distributions"]["reference"].get(tag, 0)
            print(f"  {tag:6s}: {s:5d} → {p:5d} → {r:5d}")


if __name__ == "__main__":
    main()
