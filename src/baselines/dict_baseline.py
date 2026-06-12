"""
Dictionary baseline: token-level mapping learned from training pairs.

For dialect→standard: map dialect tokens to their most frequent standard equivalent.
For standard→dialect: map standard tokens to their most frequent dialect equivalent
  per region (reverse dictionary).

Uses longest-match-first to handle multi-syllable expressions like "anh ấy" → "ảnh".
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.model.config import DataConfig

cfg = DataConfig()


def build_token_map(
    train_path: Path,
    direction: str = "forward",
) -> dict[str, dict[str, Counter]]:
    """Build token alignment dictionaries from parallel training pairs.

    Returns: {region_or_global: {source_token: Counter({target_token: count})}}

    For forward (dialect→std): source=dialect, target=standard, key="global"
    For reverse (std→dialect):  source=standard, target=dialect, key=region
    """
    maps: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    with open(train_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if direction == "forward" and r["task"] != "dialect2std":
                continue
            if direction == "reverse" and not r["task"].startswith("std2dialect"):
                continue
            if direction == "lexnorm" and r["task"] != "lexnorm":
                continue

            src_tokens = r["source"].lower().split()
            tgt_tokens = r["target"].lower().split()
            key = (r.get("region") or "global") if direction == "reverse" else "global"

            # Simple alignment: pair tokens at same position
            for i, (s, t) in enumerate(zip(src_tokens, tgt_tokens)):
                if s != t:
                    maps[key][s][t] += 1

            # Also capture bigrams for multi-syllable mappings
            for i in range(len(src_tokens) - 1):
                bigram_src = src_tokens[i] + " " + src_tokens[i + 1]
                if i < len(tgt_tokens):
                    maps[key][bigram_src][tgt_tokens[i]] += 1

    # Flatten to most-common mapping
    result: dict[str, dict[str, str]] = {}
    for key, token_map in maps.items():
        result[key] = {}
        for src_tok, counter in token_map.items():
            best_target = counter.most_common(1)[0][0]
            result[key][src_tok] = best_target

    return result


def apply_dictionary(
    text: str,
    token_map: dict[str, str],
) -> str:
    """Apply longest-match-first token replacement."""
    words = text.lower().split()
    output = []
    i = 0
    while i < len(words):
        # Try bigram match first
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in token_map:
                output.append(token_map[bigram])
                i += 2
                continue
        # Unigram match
        if words[i] in token_map:
            output.append(token_map[words[i]])
        else:
            output.append(words[i])
        i += 1
    return " ".join(output)


def predict(
    source: str,
    token_map: dict[str, str],
) -> str:
    return apply_dictionary(source, token_map)


def run_on_file(
    test_path: Path,
    train_path: Path,
    output_path: Path,
    tasks: list[str] | None = None,
    direction: str = "forward",
):
    """Run dictionary baseline on a test file."""
    maps = build_token_map(train_path, direction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(test_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if tasks and record["task"] not in tasks:
                continue

            region = record.get("region") or "global"
            key = region if direction == "reverse" else "global"
            token_map = maps.get(key, {})

            results.append({
                "task": record["task"],
                "region": record.get("region"),
                "source": record["source"],
                "target": record["target"],
                "prediction": predict(record["source"], token_map),
                "baseline": "dictionary",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Dictionary baseline ({direction}): {len(results)} predictions → {output_path}")
    return results


if __name__ == "__main__":
    train = cfg.processed_dir / "train.jsonl"
    test = cfg.processed_dir / "test.jsonl"
    out_dir = cfg.processed_dir.parent.parent / "results" / "metrics"

    # Forward direction
    run_on_file(test, train, out_dir / "dict_fwd_predictions.jsonl",
                tasks=["dialect2std"], direction="forward")
    # Reverse direction
    run_on_file(test, train, out_dir / "dict_rev_predictions.jsonl",
                tasks=["std2dialect_northern", "std2dialect_central", "std2dialect_southern"],
                direction="reverse")
