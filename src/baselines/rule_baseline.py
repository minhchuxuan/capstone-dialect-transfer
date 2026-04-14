"""
Rule-based baseline for Standard → Dialect and Dialect → Standard.

Hand-crafted rules per region covering:
  - Lexical substitutions (đâu↔mô, gì↔chi, etc.)
  - Pronoun contractions (anh ấy→ảnh, bà ấy→bả)
  - Sentence-final particles

This baseline is strong for known patterns but fails on unseen vocab.
It demonstrates NLP fundamentals: morphological awareness, pattern matching.
"""
import json
import re
from pathlib import Path

from src.model.config import DataConfig

cfg = DataConfig()

# ---------------------------------------------------------------------------
# Rule tables per region
# ---------------------------------------------------------------------------

# Rules for Standard → Central dialect
STD_TO_CENTRAL = [
    # Multi-word first (longer patterns match before shorter)
    (r"\banh ấy\b", "ảnh"),
    (r"\bchị ấy\b", "chỉ"),
    (r"\bbà ấy\b", "bả"),
    (r"\bông ấy\b", "ổng"),
    (r"\bcô ấy\b", "cổ"),
    (r"\bthế nào\b", "răng"),
    (r"\bnhư thế\b", "rứa"),
    (r"\bnhư vậy\b", "rứa"),
    (r"\btại sao\b", "răng"),
    # Single-word
    (r"\bđâu\b", "mô"),
    (r"\bgì\b", "chi"),
    (r"\bsao\b", "răng"),
    (r"\bvậy\b", "rứa"),
    (r"\bkhông\b", "khôn"),
    (r"\bnào\b", "mô"),
    (r"\bđây\b", "ni"),
    (r"\bđó\b", "tê"),
    (r"\bkia\b", "tề"),
    (r"\brồi\b", "rồi"),
    (r"\bbây giờ\b", "chừ"),
]

# Rules for Standard → Southern dialect
STD_TO_SOUTH = [
    (r"\banh ấy\b", "ảnh"),
    (r"\bchị ấy\b", "chỉ"),
    (r"\bbà ấy\b", "bả"),
    (r"\bông ấy\b", "ổng"),
    (r"\bcô ấy\b", "cổ"),
    (r"\bthế nào\b", "sao"),
    (r"\bnhư thế\b", "vậy đó"),
    (r"\btại sao\b", "sao"),
    (r"\bkhông\b", "hông"),
    (r"\bvậy\b", "vậy"),
    (r"\brồi\b", "rồi"),
    (r"\bbiết\b", "biết"),
    (r"\bđây\b", "đây nè"),
    (r"\bnhé\b", "nghen"),
    (r"\bnhỉ\b", "hen"),
]

# Rules for Standard → Northern non-standard
STD_TO_NORTH = [
    (r"\banh ấy\b", "anh ý"),
    (r"\bnó\b", "nó"),
    (r"\btrời ơi\b", "giời ơi"),
    (r"\btrời\b", "giời"),
    (r"\bthế\b", "thế"),
    (r"\bnhỉ\b", "nhở"),
    (r"\bnhé\b", "nhá"),
]

# Reverse rules for Dialect → Standard (Central)
CENTRAL_TO_STD = [
    (r"\bảnh\b", "anh ấy"),
    (r"\bchỉ\b", "chị ấy"),
    (r"\bbả\b", "bà ấy"),
    (r"\bổng\b", "ông ấy"),
    (r"\bcổ\b", "cô ấy"),
    (r"\bmô\b", "đâu"),
    (r"\bchi\b", "gì"),
    (r"\brăng\b", "sao"),
    (r"\brứa\b", "vậy"),
    (r"\bkhôn\b", "không"),
    (r"\bni\b", "đây"),
    (r"\btê\b", "đó"),
    (r"\btề\b", "kia"),
    (r"\bchừ\b", "bây giờ"),
]

# Reverse rules for Dialect → Standard (Southern)
SOUTH_TO_STD = [
    (r"\bảnh\b", "anh ấy"),
    (r"\bchỉ\b", "chị ấy"),
    (r"\bbả\b", "bà ấy"),
    (r"\bổng\b", "ông ấy"),
    (r"\bcổ\b", "cô ấy"),
    (r"\bhông\b", "không"),
    (r"\bnghen\b", "nhé"),
    (r"\bhen\b", "nhỉ"),
    (r"\bnè\b", "này"),
]

# Reverse rules for Dialect → Standard (Northern non-standard)
NORTH_TO_STD = [
    (r"\banh ý\b", "anh ấy"),
    (r"\bgiời\b", "trời"),
    (r"\bnhở\b", "nhỉ"),
    (r"\bnhá\b", "nhé"),
]

# ---------------------------------------------------------------------------
# Rule lookup
# ---------------------------------------------------------------------------

RULE_TABLES = {
    # Standard → Dialect
    "std2dialect_central": STD_TO_CENTRAL,
    "std2dialect_south": STD_TO_SOUTH,
    "std2dialect_north": STD_TO_NORTH,
    # Dialect → Standard
    "dialect2std_central": CENTRAL_TO_STD,
    "dialect2std_south": SOUTH_TO_STD,
    "dialect2std_north": NORTH_TO_STD,
}


def apply_rules(text: str, rules: list[tuple[str, str]]) -> str:
    """Apply regex rules in order (longest patterns first)."""
    result = text
    for pattern, replacement in rules:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def predict(source: str, task: str, region: str | None = None) -> str:
    """Apply rule-based transformation."""
    if task == "dialect2std" and region:
        key = f"dialect2std_{region}"
    elif task.startswith("std2dialect") and region:
        key = f"std2dialect_{region}"
    elif task.startswith("std2dialect"):
        # Extract region from task name
        key = task
    else:
        return source  # no rules for this task

    rules = RULE_TABLES.get(key, [])
    if not rules:
        return source
    return apply_rules(source, rules)


def run_on_file(
    test_path: Path,
    output_path: Path,
    tasks: list[str] | None = None,
):
    """Run rule-based baseline on a test file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(test_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if tasks and record["task"] not in tasks:
                continue
            results.append({
                "task": record["task"],
                "region": record.get("region"),
                "source": record["source"],
                "target": record["target"],
                "prediction": predict(
                    record["source"], record["task"], record.get("region")
                ),
                "baseline": "rule_based",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Rule-based baseline: {len(results)} predictions → {output_path}")
    return results


if __name__ == "__main__":
    test = cfg.processed_dir / "test.jsonl"
    out_dir = cfg.processed_dir.parent.parent / "results" / "metrics"
    # All tasks
    run_on_file(test, out_dir / "rule_predictions.jsonl")
