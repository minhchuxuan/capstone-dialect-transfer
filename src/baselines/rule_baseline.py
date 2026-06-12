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
    # Multi-word pronoun contractions first
    (r"\banh ấy\b", "ảnh"),
    (r"\bchị ấy\b", "chỉ"),
    (r"\bbà ấy\b", "bả"),
    (r"\bông ấy\b", "ổng"),
    (r"\bcô ấy\b", "cổ"),
    (r"\bở trên\b", "trển"),
    # Negation (southern hông/hổng)
    (r"\bkhông\b", "hông"),
    (r"\bchẳng\b", "hổng"),
    # Kinship terms
    (r"\bbố\b", "ba"),
    (r"\bmẹ\b", "má"),
    # Sentence-final particles
    (r"\bnhé\b", "nghen"),
    (r"\bnhỉ\b", "hen"),
]

# Rules for Standard → Northern non-standard
STD_TO_NORTH = [
    # High-precision phonological/lexical northern colloquialisms
    (r"\btrời ơi\b", "giời ơi"),
    (r"\btrời\b", "giời"),
    (r"\bnhỉ\b", "nhở"),
    (r"\bnhé\b", "nhá"),
    (r"\bvâng\b", "dạ"),
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
    (r"\btrển\b", "ở trên"),
    (r"\bhổng\b", "không"),
    (r"\bhông\b", "không"),
    (r"\bnghen\b", "nhé"),
    (r"\bnhen\b", "nhé"),
    (r"\bhen\b", "nhỉ"),
    (r"\bnè\b", "này"),
]

# Reverse rules for Dialect → Standard (Northern non-standard)
NORTH_TO_STD = [
    (r"\bgiời ơi\b", "trời ơi"),
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
    "std2dialect_southern": STD_TO_SOUTH,
    "std2dialect_northern": STD_TO_NORTH,
    # Dialect → Standard
    "dialect2std_central": CENTRAL_TO_STD,
    "dialect2std_southern": SOUTH_TO_STD,
    "dialect2std_northern": NORTH_TO_STD,
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
