"""
ChelatedAI Configuration Module

Centralized configuration management for hyperparameters, paths, and system settings.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any
import json


# ===== Security Utilities =====

def validate_safe_path(path: Path, base_dir: Optional[Path] = None, allow_absolute: bool = True) -> Path:
    """
    Validate that a path is safe from path traversal attacks.
    
    Args:
        path: Path to validate
        base_dir: Optional base directory to restrict paths to
        allow_absolute: If False, reject absolute paths (default: True for backwards compatibility)
    
    Returns:
        Resolved safe path
        
    Raises:
        ValueError: If path contains traversal attempts or is outside base_dir
    """
    # Convert to Path object
    path = Path(path)
    
    # Check for path traversal components before resolution
    parts = path.parts
    if '..' in parts:
        raise ValueError("Path traversal detected: path contains '..' components")
    
    # Resolve to absolute path
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {e}")
    
    # If base_dir specified, ensure resolved path is within it
    if base_dir is not None:
        base_dir = Path(base_dir).resolve()
        try:
            resolved_path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path escapes base directory: {path} not under {base_dir}")
    
    return resolved_path


def sanitize_name(name: str, pattern: str = r'^[a-zA-Z0-9_-]+$') -> str:
    """
    Sanitize a name using allowlist pattern.
    
    Args:
        name: Name to sanitize
        pattern: Regex pattern for allowed characters (default: alphanumeric, underscore, hyphen)
    
    Returns:
        Sanitized name
        
    Raises:
        ValueError: If name doesn't match allowed pattern
    """
    if not name:
        raise ValueError("Name cannot be empty")
    
    if not re.match(pattern, name):
        raise ValueError(
            f"Invalid name: '{name}' contains disallowed characters. "
            f"Only alphanumeric, underscore, and hyphen allowed."
        )
    
    return name


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
    OLLAMA_INPUT_MAX_CHARS = 10000  # Hard safety cap before truncation retries

    # Truncation strategy for long documents (Ollama mode)
    OLLAMA_TRUNCATION_LIMITS = [6000, 2000, 500]  # chars, tried in order

    # ===== Chelation Hyperparameters =====
    # Core chelation parameters
    DEFAULT_CHELATION_P = 85  # Percentile threshold for dimension selection (0-100)
    DEFAULT_CHELATION_THRESHOLD = 0.0004  # Variance threshold for adaptive triggering
    
    # ===== Adaptive Threshold Configuration =====
    # Adaptive threshold tuning (opt-in feature)
    ADAPTIVE_THRESHOLD_ENABLED = False  # Disabled by default for backward compatibility
    ADAPTIVE_THRESHOLD_PERCENTILE = 75  # Target percentile of observed variances
    ADAPTIVE_THRESHOLD_WINDOW = 100  # Number of recent variance samples to track
    ADAPTIVE_THRESHOLD_MIN_SAMPLES = 20  # Minimum samples before adaptive adjustment
    ADAPTIVE_THRESHOLD_MIN = 0.0001  # Safety lower bound for threshold
    ADAPTIVE_THRESHOLD_MAX = 0.01  # Safety upper bound for threshold

    # ===== Noise Injection (Experimental) =====
    NOISE_INJECTION_ENABLED = False
    NOISE_INJECTION_BASE_SCALE = 0.05
    NOISE_INJECTION_MAX_SCALE = 0.5

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
    
    # ===== Payload Optimization (F-040) =====
    # Control whether to store/fetch full document text in Qdrant payload
    STORE_FULL_TEXT_PAYLOAD = True  # Store text in payload during ingestion (default: True for backward compatibility)
    FETCH_PAYLOAD_ON_QUERY = False  # Fetch payload during query operations where not needed (default: False for optimization)

    # ===== Convergence Detection (Phase 1) =====
    CONVERGENCE_ENABLED = False  # Opt-in early stopping for training loops
    CONVERGENCE_PATIENCE = 5  # Epochs without improvement before stopping
    CONVERGENCE_REL_THRESHOLD = 0.001  # Minimum relative improvement to count
    CONVERGENCE_MIN_EPOCHS = 3  # Minimum epochs before early stopping triggers

    CONVERGENCE_PRESETS = {
        "patient": {
            "patience": 10,
            "rel_threshold": 0.0005,
            "min_epochs": 5,
            "description": "Patient convergence, longer training"
        },
        "balanced": {
            "patience": 5,
            "rel_threshold": 0.001,
            "min_epochs": 3,
            "description": "Standard convergence detection"
        },
        "aggressive": {
            "patience": 2,
            "rel_threshold": 0.005,
            "min_epochs": 2,
            "description": "Quick convergence, stop early"
        }
    }

    # ===== Temperature Scaling (Phase 1) =====
    TEMPERATURE_SCALING_ENABLED = False  # Opt-in temperature for spectral chelation
    DEFAULT_TEMPERATURE = 1.0  # Temperature divisor for chelation scores (1.0 = no effect)

    # ===== Adapter Type Selection (Phase 2) =====
    ADAPTER_TYPE = "mlp"  # "mlp", "procrustes", or "low_rank"
    LOW_RANK_ADAPTER_RANK = 16  # Rank for low-rank affine adapter

    ADAPTER_TYPE_PRESETS = {
        "mlp": {
            "adapter_type": "mlp",
            "description": "Original MLP residual adapter"
        },
        "procrustes": {
            "adapter_type": "procrustes",
            "description": "Orthogonal Procrustes via Cayley parameterization"
        },
        "low_rank": {
            "adapter_type": "low_rank",
            "rank": 16,
            "description": "Low-rank affine correction"
        }
    }

    # ===== Online Updates (Phase 3) =====
    ONLINE_UPDATE_ENABLED = False  # Opt-in inference-time adapter updates
    ONLINE_LEARNING_RATE = 0.0001
    ONLINE_MICRO_STEPS = 1
    ONLINE_MOMENTUM = 0.9
    ONLINE_MAX_GRAD_NORM = 1.0
    ONLINE_UPDATE_INTERVAL = 1

    # ===== Learned Dimension Masking (Phase 4) =====
    LEARNED_MASK_ENABLED = False  # Opt-in neural mask predictor
    MASK_PREDICTOR_HIDDEN_RATIO = 0.25
    MASK_PREDICTOR_THRESHOLD = 0.5

    # ===== Per-Embedding Quality Assessment (Phase 4) =====
    QUALITY_ASSESSMENT_ENABLED = False  # Opt-in per-doc quality scores
    QUALITY_DECAY_FACTOR = 0.95
    QUALITY_HIGH_THRESHOLD = 0.8
    QUALITY_LOW_THRESHOLD = 0.3

    # ===== Dimension Projection Configuration =====
    PROJECTION_ENABLED = True  # Auto-project when teacher/student dims mismatch
    PROJECTION_HIDDEN_DIM = None  # None = direct projection, int = bottleneck

    # ===== Adapter Training Configuration =====
    # Session 21: 81-config SciFact NDCG@10 sweep validated these defaults.
    # LR>=0.1 causes catastrophic collapse (63% of configs). LR=0.01 is safest.
    # Threshold=1 preserves quality; threshold>=2 causes severe degradation.
    # Push magnitude 0.05 reduces degradation ~29% vs 0.1.
    DEFAULT_LEARNING_RATE = 0.01  # Sweep-validated: LR=0.01 is safest (Session 21, SciFact NDCG@10)
    DEFAULT_EPOCHS = 10
    DEFAULT_COLLAPSE_THRESHOLD = 1  # Sweep-validated: threshold=1 preserves quality; >=2 degrades (Session 21)
    ADAPTER_HIDDEN_DIM_RATIO = 0.5  # hidden_dim = input_dim * ratio
    HOMEOSTATIC_PUSH_MAGNITUDE = 0.05  # Sweep-validated: 0.05 reduces degradation ~29% vs 0.1 (Session 21)
    
    # ===== Teacher Distillation Configuration =====
    # Training mode: 'baseline', 'offline', 'hybrid'
    DEFAULT_TRAINING_MODE = "baseline"
    
    # Teacher model for distillation (local sentence-transformers model)
    DEFAULT_TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Teacher weight for hybrid mode (0.0 = homeostatic only, 1.0 = teacher only)
    DEFAULT_TEACHER_WEIGHT = 0.5
    
    # Offline distillation epochs
    DEFAULT_OFFLINE_EPOCHS = 15
    DEFAULT_OFFLINE_LEARNING_RATE = 0.005

    # Adapter training tuning by dataset size
    # Updated Session 21: collapse thresholds lowered based on sweep analysis.
    # Sweep showed threshold>=2 causes degradation; LR>=0.1 causes catastrophic collapse.
    ADAPTER_PRESETS = {
        "small": {  # <1000 documents
            "learning_rate": 0.01,
            "epochs": 5,
            "threshold": 3,
            "description": "Few documents, moderate threshold (sweep-safe LR)"
        },
        "medium": {  # 1000-10000 documents
            "learning_rate": 0.005,
            "epochs": 10,
            "threshold": 1,
            "description": "Standard datasets, sweep-validated threshold"
        },
        "large": {  # >10000 documents
            "learning_rate": 0.001,
            "epochs": 15,
            "threshold": 1,
            "description": "Large datasets, capture all patterns"
        },
        "sweep_optimal": {
            "learning_rate": 0.01,
            "epochs": 5,
            "threshold": 1,
            "description": "Best config from 81-config SciFact NDCG@10 sweep (Session 21)"
        }
    }
    
    # RLM (Recursive Logic Mining) presets
    RLM_PRESETS = {
        "balanced": {
            "max_depth": 3,
            "min_support": 5,
            "description": "Balanced depth for general decomposition"
        },
        "shallow": {
            "max_depth": 2,
            "min_support": 10,
            "description": "Shallow decomposition for simple queries"
        },
        "deep": {
            "max_depth": 5,
            "min_support": 2,
            "description": "Deep decomposition for complex queries"
        }
    }
    
    # Ensemble teacher presets
    ENSEMBLE_PRESETS = {
        "diverse": {
            "models": [
                ("all-MiniLM-L6-v2", 0.4),
                ("all-mpnet-base-v2", 0.6),
            ],
            "description": "Diverse model architectures for robust distillation",
        },
        "multilingual": {
            "models": [
                ("all-MiniLM-L6-v2", 0.5),
                ("paraphrase-multilingual-MiniLM-L12-v2", 0.5),
            ],
            "description": "Multilingual ensemble for cross-lingual transfer",
        },
    }

    # Sedimentation presets
    # Updated Session 21: 81-config SciFact sweep showed threshold>=2 causes severe degradation.
    # All presets now use safer values validated by parameter sweep analysis.
    SEDIMENTATION_PRESETS = {
        "balanced": {
            "collapse_threshold": 1,
            "push_magnitude": 0.05,
            "description": "Sweep-validated safe defaults (Session 21: threshold=1, push=0.05)"
        },
        "conservative": {
            "collapse_threshold": 3,
            "push_magnitude": 0.05,
            "description": "Conservative sedimentation, higher threshold tolerates more collapse"
        },
        "aggressive": {
            "collapse_threshold": 1,
            "push_magnitude": 0.1,
            "description": "Aggressive push magnitude for noisy data (sweep: 0.1 still safe at threshold=1)"
        },
        "sweep_optimal": {
            "collapse_threshold": 1,
            "push_magnitude": 0.05,
            "noise_injection_base_scale": 0.05,
            "learning_rate": 0.01,
            "description": "Best config from 81-config SciFact NDCG@10 sweep (Session 21)"
        }
    }

    # Teacher weight schedule presets
    TEACHER_WEIGHT_SCHEDULE_PRESETS = {
        "constant": {
            "schedule": "constant",
            "initial_weight": 0.5,
            "description": "Fixed teacher weight throughout training",
        },
        "gradual_decay": {
            "schedule": "linear_decay",
            "initial_weight": 0.7,
            "final_weight": 0.1,
            "total_steps": 100,
            "description": "Gradual linear decrease in teacher influence",
        },
        "cosine": {
            "schedule": "cosine_annealing",
            "initial_weight": 0.6,
            "final_weight": 0.05,
            "total_steps": 100,
            "description": "Smooth cosine decay of teacher influence",
        },
        "aggressive_decay": {
            "schedule": "step_decay",
            "initial_weight": 0.8,
            "gamma": 0.5,
            "step_size": 20,
            "description": "Aggressive step-wise decay of teacher influence",
        },
        "adaptive": {
            "schedule": "adaptive",
            "initial_weight": 0.5,
            "patience": 5,
            "description": "Loss-adaptive teacher weight adjustment",
        },
    }

    # ===== Memory Management =====
    MAX_BATCH_MEMORY_MB = 512  # Target max memory per batch
    CHUNK_SIZE = 100  # Qdrant update chunk size
    
    # Streaming ingestion parameters
    STREAMING_BATCH_SIZE = 100  # Documents per batch for streaming ingestion
    STREAMING_PROGRESS_INTERVAL = 10  # Log progress every N batches
    
    # Chelation log memory management
    CHELATION_LOG_MAX_ENTRIES_PER_DOC = 1000  # Max entries per document in chelation log

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
    MIN_MAX_DEPTH = 1
    MAX_MAX_DEPTH = 10

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
        if value < 0:
            print(f"WARNING: epochs={value} is negative, using 0 (skip training).")
            return 0
        if not cls.MIN_EPOCHS <= value <= cls.MAX_EPOCHS:
            print(f"WARNING: epochs={value} out of range [{cls.MIN_EPOCHS}, {cls.MAX_EPOCHS}], clamping.")
            return max(cls.MIN_EPOCHS, min(cls.MAX_EPOCHS, value))
        return value
    
    @classmethod
    def validate_training_mode(cls, value: str) -> str:
        """Validate training mode."""
        valid_modes = ["baseline", "offline", "hybrid"]
        if value not in valid_modes:
            print(f"WARNING: training_mode='{value}' invalid. Valid options: {valid_modes}. Using 'baseline'.")
            return "baseline"
        return value
    
    @classmethod
    def validate_teacher_weight(cls, value: float) -> float:
        """Validate and clamp teacher_weight to [0.0, 1.0]."""
        if not 0.0 <= value <= 1.0:
            print(f"WARNING: teacher_weight={value} out of range [0.0, 1.0], clamping.")
            return max(0.0, min(1.0, value))
        return value
    
    @classmethod
    def validate_adaptive_percentile(cls, value: float) -> float:
        """Validate and clamp adaptive threshold percentile to [0, 100]."""
        if not 0.0 <= value <= 100.0:
            print(f"WARNING: adaptive_percentile={value} out of range [0.0, 100.0], clamping.")
            return max(0.0, min(100.0, value))
        return value
    
    @classmethod
    def validate_adaptive_window(cls, value: int) -> int:
        """Validate adaptive threshold window size."""
        if value < 1:
            print(f"WARNING: adaptive_window={value} must be >= 1, using 1.")
            return 1
        return value
    
    @classmethod
    def validate_adaptive_min_samples(cls, value: int) -> int:
        """Validate adaptive threshold minimum samples."""
        if value < 1:
            print(f"WARNING: adaptive_min_samples={value} must be >= 1, using 1.")
            return 1
        return value
    
    @classmethod
    def validate_max_depth(cls, value: int) -> int:
        """Validate and clamp max_depth to valid range."""
        if not cls.MIN_MAX_DEPTH <= value <= cls.MAX_MAX_DEPTH:
            print(f"WARNING: max_depth={value} out of range [{cls.MIN_MAX_DEPTH}, {cls.MAX_MAX_DEPTH}], clamping.")
            return max(cls.MIN_MAX_DEPTH, min(cls.MAX_MAX_DEPTH, value))
        return value

    @classmethod
    def get_preset(cls, preset_name: str, preset_type: str = "chelation") -> Dict[str, Any]:
        """
        Get predefined configuration preset.

        Args:
            preset_name: Name of preset ('conservative', 'balanced', 'aggressive', etc.)
            preset_type: Type of preset ('chelation', 'adapter', 'rlm', 'sedimentation')

        Returns:
            Dictionary with preset parameters

        Raises:
            ValueError: If preset not found or preset_type invalid
        """
        # Map preset_type to preset dictionary
        preset_map = {
            "chelation": cls.CHELATION_PRESETS,
            "adapter": cls.ADAPTER_PRESETS,
            "rlm": cls.RLM_PRESETS,
            "sedimentation": cls.SEDIMENTATION_PRESETS,
            "convergence": cls.CONVERGENCE_PRESETS,
            "adapter_type": cls.ADAPTER_TYPE_PRESETS,
            "ensemble": cls.ENSEMBLE_PRESETS,
            "teacher_weight_schedule": cls.TEACHER_WEIGHT_SCHEDULE_PRESETS,
        }
        
        if preset_type not in preset_map:
            valid_types = ", ".join(preset_map.keys())
            raise ValueError(f"Invalid preset_type '{preset_type}'. Valid types: {valid_types}")
        
        presets = preset_map[preset_type]

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
        # Validate path for traversal attacks
        config_path = validate_safe_path(Path(config_path))
        
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
        # Validate path for traversal attacks
        config_path = validate_safe_path(Path(config_path))
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
