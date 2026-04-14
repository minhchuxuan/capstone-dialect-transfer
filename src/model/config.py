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

    # Task prefixes — the model sees these as the first tokens of input
    task_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "dialect2std": "dialect2std",
        "std2dialect_north": "std2dialect_north",
        "std2dialect_central": "std2dialect_central",
        "std2dialect_south": "std2dialect_south",
        "lexnorm": "lexnorm",
        "spell": "spell",
    })

    # Multi-task mixing ratios (weights for sampling during training)
    task_ratios: Dict[str, float] = field(default_factory=lambda: {
        "dialect2std": 0.30,
        "std2dialect": 0.35,
        "lexnorm": 0.25,
        "spell": 0.10,
    })

    # Region mapping from ViDia2Std labels
    region_map: Dict[str, str] = field(default_factory=lambda: {
        "Bắc": "north",
        "Trung": "central",
        "Nam": "south",
        "North": "north",
        "Central": "central",
        "South": "south",
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

    # Dialect markers for DFR metric
    dialect_markers: Dict[str, list] = field(default_factory=lambda: {
        "central": ["mô", "chi", "răng", "rứa", "nờ", "hè", "ni", "tê",
                     "ảnh", "chỉ", "bả", "ổng", "khôn"],
        "south": ["hông", "nè", "hen", "á", "nghen", "dzậy", "trển",
                   "ảnh", "chỉ", "bả", "ổng", "vậy đó", "đó"],
        "north": ["ý", "giời", "mờ", "nhở", "ấy nhở", "ơi giời"],
    })
