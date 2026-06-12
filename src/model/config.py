"""
Centralized configuration for the dialect transfer project.
All hyperparameters and paths in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Project root (relative to this file: src/model/config.py -> ../../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class ModelConfig:
    """Hyperparameters for seq2seq training."""

    # Model
    model_name: str = "vinai/bartpho-syllable"
    max_source_length: int = 128
    max_target_length: int = 128

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 3e-5
    num_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True

    # Decoding
    num_beams: int = 4
    max_new_tokens: int = 128

    # Back-translation generation (use sampling for diversity)
    bt_do_sample: bool = True
    bt_temperature: float = 1.0
    bt_top_k: int = 50
    bt_top_p: float = 0.95

    # Paths
    output_dir: str = str(RESULTS_DIR / "checkpoints")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """Paths and task configuration."""

    raw_dir: Path = DATA_DIR / "raw"
    processed_dir: Path = DATA_DIR / "processed"
    augmented_dir: Path = DATA_DIR / "augmented"
    monolingual_dir: Path = DATA_DIR / "monolingual"

    # Task prefixes — INFORMATIONAL ONLY. format_input uses the raw `task`
    # field from each record verbatim; these entries just document the
    # canonical task names that appear in the processed data.
    task_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "dialect2std": "dialect2std",
        "std2dialect_northern": "std2dialect_northern",
        "std2dialect_central": "std2dialect_central",
        "std2dialect_southern": "std2dialect_southern",
    })

    # Multi-task mixing ratios — INFORMATIONAL ONLY. Actual task mixing is done
    # by physical oversampling during the data build, NOT by sampling here.
    # Keys are the real per-task names present in the processed data.
    task_ratios: Dict[str, float] = field(default_factory=lambda: {
        "dialect2std": 0.30,
        "std2dialect_central": 0.25,
        "std2dialect_southern": 0.25,
        "std2dialect_northern": 0.20,
    })

    # Region mapping from source labels → canonical regions.
    # Robust to any casing/language of the 3 regions so that normalize_region
    # is idempotent (northern/central/southern map to themselves).
    region_map: Dict[str, str] = field(default_factory=lambda: {
        # Canonical english self-maps (idempotency)
        "northern": "northern",
        "central": "central",
        "southern": "southern",
        # Short english aliases
        "north": "northern",
        "south": "southern",
        # Vietnamese labels
        "Bắc": "northern",
        "Trung": "central",
        "Nam": "southern",
        # Capitalized english labels
        "North": "northern",
        "Central": "central",
        "South": "southern",
    })

    # Back-translation quality filters
    bt_round_trip_bleu_threshold: float = 0.5
    bt_length_ratio_max: float = 0.3
    bt_min_dialect_markers: int = 1


@dataclass
class EvalConfig:
    """Evaluation settings."""

    human_eval_samples_per_region: int = 50
    error_analysis_total: int = 50

    # Dialect markers for the Dialect Feature Recall (DFR) metric.
    # Each list is a set of real, high-signal lexical markers for the region.
    dialect_markers: Dict[str, list] = field(default_factory=lambda: {
        "central": [
            "mô", "chi", "răng", "rứa", "nớ", "tê", "ni", "hè", "nờ",
            "ả", "eng", "mi", "bây", "tui", "dừ", "chừ", "mạ", "nỏ",
            "cẳng", "chộ", "ốt dột", "khôn", "bọ", "o",
            "ảnh", "chỉ", "bả", "ổng",
        ],
        "southern": [
            "hông", "hổng", "nè", "nghen", "hen", "dzậy", "vậy", "trển",
            "tui", "ổng", "bả", "ảnh", "chỉ", "cổ", "kêu", "mắc gì",
            "hôn", "nhen", "á",
        ],
        "northern": [
            "giời", "nhở", "nhá", "thầy", "u", "bu", "nhõn", "đấy",
            "cơ", "ý", "a ri", "chi", "tầm",
        ],
    })
