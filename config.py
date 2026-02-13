"""
ChelatedAI Configuration Module

Centralized configuration management for hyperparameters, paths, and system settings.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ChelationConfig:
    """
    Configuration class for AntigravityEngine.

    All configuration parameters are centralized here to avoid magic numbers
    and improve maintainability.
    """

    # ===== Path Configuration =====
    # Use pathlib for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # Default database locations (can be overridden)
    DEFAULT_DB_PATH = PROJECT_ROOT / "db_default"
    ADAPTER_WEIGHTS_PATH = PROJECT_ROOT / "adapter_weights.pt"
    EVENT_LOG_PATH = PROJECT_ROOT / "chelation_events.jsonl"

    # ===== Embedding Configuration =====
    DEFAULT_VECTOR_SIZE = 768  # Default for most embedding models
    DEFAULT_MODEL_NAME = "ollama:nomic-embed-text"
    OLLAMA_URL = "http://localhost:11434/api/embeddings"
    OLLAMA_TIMEOUT = 30  # seconds per request
    OLLAMA_MAX_WORKERS = 2  # Concurrent requests to avoid overwhelming server

    # Truncation strategy for long documents (Ollama mode)
    OLLAMA_TRUNCATION_LIMITS = [6000, 2000, 500]  # chars, tried in order

    # ===== Chelation Hyperparameters =====
    # Core chelation parameters
    DEFAULT_CHELATION_P = 85  # Percentile threshold for dimension selection (0-100)
    DEFAULT_CHELATION_THRESHOLD = 0.0004  # Variance threshold for adaptive triggering

    # Chelation tuning guidelines by use case
    CHELATION_PRESETS = {
        "conservative": {
            "chelation_p": 95,
            "chelation_threshold": 0.0002,
            "description": "High-quality embeddings, minimal intervention"
        },
        "balanced": {
            "chelation_p": 85,
            "chelation_threshold": 0.0004,
            "description": "General purpose, tested on SciFact"
        },
        "aggressive": {
            "chelation_p": 75,
            "chelation_threshold": 0.0008,
            "description": "Lower-quality embeddings, strong noise reduction"
        }
    }

    # Collection name
    DEFAULT_COLLECTION_NAME = "antigravity_stage8"

    # Ollama context window hint
    OLLAMA_NUM_CTX = 4096

    # ===== Retrieval Configuration =====
    SCOUT_K = 50  # Neighborhood size for variance calculation
    TOP_K = 10  # Number of results to return
    BATCH_SIZE = 100  # Documents per ingestion batch

    # ===== Adapter Training Configuration =====
    DEFAULT_LEARNING_RATE = 0.001  # Conservative by default
    DEFAULT_EPOCHS = 10
    DEFAULT_COLLAPSE_THRESHOLD = 3  # Min frequency to trigger sedimentation
    ADAPTER_HIDDEN_DIM_RATIO = 0.5  # hidden_dim = input_dim * ratio
    HOMEOSTATIC_PUSH_MAGNITUDE = 0.1  # Adapter push magnitude for sedimentation target vectors

    # Adapter training tuning by dataset size
    ADAPTER_PRESETS = {
        "small": {  # <1000 documents
            "learning_rate": 0.01,
            "epochs": 5,
            "threshold": 10,
            "description": "Few documents, require strong signal"
        },
        "medium": {  # 1000-10000 documents
            "learning_rate": 0.005,
            "epochs": 10,
            "threshold": 3,
            "description": "Standard datasets"
        },
        "large": {  # >10000 documents
            "learning_rate": 0.001,
            "epochs": 15,
            "threshold": 1,
            "description": "Large datasets, capture all patterns"
        }
    }

    # ===== Memory Management =====
    MAX_BATCH_MEMORY_MB = 512  # Target max memory per batch
    CHUNK_SIZE = 100  # Qdrant update chunk size

    # ===== Quantization Settings =====
    QUANTIZATION_TYPE = "INT8"  # Qdrant scalar quantization
    QUANTIZATION_QUANTILE = 0.99
    QUANTIZATION_ALWAYS_RAM = True

    # ===== Logging Configuration =====
    LOG_ENCODING = "utf-8"
    LOG_QUERY_SNIPPET_LENGTH = 50

    # ===== Validation & Safety =====
    MIN_CHELATION_P = 0
    MAX_CHELATION_P = 100
    MIN_LEARNING_RATE = 0.0001
    MAX_LEARNING_RATE = 1.0
    MIN_EPOCHS = 1
    MAX_EPOCHS = 100

    @classmethod
    def validate_chelation_p(cls, value: float) -> float:
        """Validate and clamp chelation_p to valid range."""
        if not cls.MIN_CHELATION_P <= value <= cls.MAX_CHELATION_P:
            print(f"WARNING: chelation_p={value} out of range [{cls.MIN_CHELATION_P}, {cls.MAX_CHELATION_P}], clamping.")
            return max(cls.MIN_CHELATION_P, min(cls.MAX_CHELATION_P, value))
        return value

    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        """Validate and clamp learning_rate to valid range."""
        if not cls.MIN_LEARNING_RATE <= value <= cls.MAX_LEARNING_RATE:
            print(f"WARNING: learning_rate={value} out of range [{cls.MIN_LEARNING_RATE}, {cls.MAX_LEARNING_RATE}], clamping.")
            return max(cls.MIN_LEARNING_RATE, min(cls.MAX_LEARNING_RATE, value))
        return value

    @classmethod
    def validate_epochs(cls, value: int) -> int:
        """Validate and clamp epochs to valid range."""
        if not cls.MIN_EPOCHS <= value <= cls.MAX_EPOCHS:
            print(f"WARNING: epochs={value} out of range [{cls.MIN_EPOCHS}, {cls.MAX_EPOCHS}], clamping.")
            return max(cls.MIN_EPOCHS, min(cls.MAX_EPOCHS, value))
        return value

    @classmethod
    def get_preset(cls, preset_name: str, preset_type: str = "chelation") -> Dict[str, Any]:
        """
        Get predefined configuration preset.

        Args:
            preset_name: Name of preset ('conservative', 'balanced', 'aggressive', etc.)
            preset_type: Type of preset ('chelation' or 'adapter')

        Returns:
            Dictionary with preset parameters

        Raises:
            ValueError: If preset not found
        """
        presets = cls.CHELATION_PRESETS if preset_type == "chelation" else cls.ADAPTER_PRESETS

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")

        return presets[preset_name].copy()

    @classmethod
    def load_from_file(cls, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config

    @classmethod
    def save_to_file(cls, config: Dict[str, Any], config_path: Path):
        """
        Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            config_path: Path to save JSON config
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def get_db_path(cls, task_name: str) -> Path:
        """
        Get cross-platform database path for a task.

        Args:
            task_name: Name of the task (e.g., 'SciFact', 'NFCorpus')

        Returns:
            Pathlib Path object for database location
        """
        db_name = f"db_{task_name.lower()}_evolution"
        return cls.PROJECT_ROOT / db_name

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.ADAPTER_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls.EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# Convenience function for quick config access
def get_config(preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration dictionary with optional preset.

    Args:
        preset: Optional preset name ('conservative', 'balanced', 'aggressive')

    Returns:
        Configuration dictionary
    """
    if preset:
        return ChelationConfig.get_preset(preset, "chelation")

    return {
        "chelation_p": ChelationConfig.DEFAULT_CHELATION_P,
        "chelation_threshold": ChelationConfig.DEFAULT_CHELATION_THRESHOLD,
        "learning_rate": ChelationConfig.DEFAULT_LEARNING_RATE,
        "epochs": ChelationConfig.DEFAULT_EPOCHS,
        "scout_k": ChelationConfig.SCOUT_K,
    }


if __name__ == "__main__":
    # Demo: Show all presets
    print("=== Chelation Presets ===")
    for name, preset in ChelationConfig.CHELATION_PRESETS.items():
        print(f"\n{name.upper()}:")
        for key, value in preset.items():
            print(f"  {key}: {value}")

    print("\n=== Adapter Presets ===")
    for name, preset in ChelationConfig.ADAPTER_PRESETS.items():
        print(f"\n{name.upper()}:")
        for key, value in preset.items():
            print(f"  {key}: {value}")

    print(f"\n=== Paths (Platform: {os.name}) ===")
    print(f"Project Root: {ChelationConfig.PROJECT_ROOT}")
    print(f"Adapter Weights: {ChelationConfig.ADAPTER_WEIGHTS_PATH}")
    print(f"Event Log: {ChelationConfig.EVENT_LOG_PATH}")
    print(f"SciFact DB: {ChelationConfig.get_db_path('SciFact')}")
