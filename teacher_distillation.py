"""
Teacher Distillation Module for ChelatedAI

Provides distillation helpers for offline teacher training and hybrid modes.
Uses local sentence-transformers model to avoid external API dependencies.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from chelation_logger import get_logger
from config import ChelationConfig

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore


class TeacherDistillationHelper:
    """
    Helper class for teacher distillation operations.
    
    Encapsulates teacher model loading, target generation, and blending logic.
    """
    
    def __init__(self, teacher_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize teacher distillation helper.
        
        Args:
            teacher_model_name: HuggingFace model name for teacher
        """
        self.logger = get_logger()
        self.teacher_model_name = teacher_model_name
        self.teacher_model = None
        self.teacher_dim = None
        
        self.logger.log_event(
            "distillation_init",
            f"Initializing teacher distillation with model: {teacher_model_name}",
            teacher_model=teacher_model_name
        )
    
    def load_teacher_model(self):
        """Load teacher model lazily (only when needed)."""
        if self.teacher_model is not None:
            return
        
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.log_event(
                "teacher_load",
                f"Loading teacher model on device: {device}",
                device=device
            )
            
            self.teacher_model = SentenceTransformer(
                self.teacher_model_name,
                device=device,
                trust_remote_code=True
            )
            self.teacher_dim = self.teacher_model.get_sentence_embedding_dimension()
            
            self.logger.log_event(
                "teacher_load_success",
                f"Teacher model loaded. Dimension: {self.teacher_dim}",
                teacher_dim=self.teacher_dim
            )
        except ImportError as e:
            self.logger.log_error(
                "teacher_load_failed",
                "sentence-transformers not available",
                exception=e
            )
            raise ImportError(
                "sentence-transformers required for teacher distillation. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            self.logger.log_error(
                "teacher_load_failed",
                f"Failed to load teacher model: {e}",
                exception=e
            )
            raise
    
    def get_teacher_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings from teacher model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of teacher embeddings [batch_size, teacher_dim]
        """
        if not texts:
            return np.array([])
        
        self.load_teacher_model()
        
        try:
            embeddings = self.teacher_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Ensure unit norm for cosine similarity
            )
            return embeddings
        except Exception as e:
            self.logger.log_error(
                "teacher_embedding_failed",
                f"Teacher embedding failed for {len(texts)} texts",
                exception=e,
                num_texts=len(texts)
            )
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.teacher_dim))
    
    def check_dimension_compatibility(self, student_dim: int) -> bool:
        """
        Check if teacher and student dimensions are compatible.
        
        Args:
            student_dim: Dimension of student embeddings
            
        Returns:
            True if compatible (same dimension), False otherwise
        """
        self.load_teacher_model()
        
        compatible = (self.teacher_dim == student_dim)
        
        if not compatible:
            self.logger.log_event(
                "dimension_mismatch",
                f"Teacher dim ({self.teacher_dim}) != Student dim ({student_dim})",
                level="WARNING",
                teacher_dim=self.teacher_dim,
                student_dim=student_dim
            )
        
        return compatible
    
    def generate_distillation_targets(
        self,
        texts: List[str],
        current_embeddings: np.ndarray,
        teacher_weight: float = 1.0
    ) -> np.ndarray:
        """
        Generate distillation targets by blending teacher and current embeddings.
        
        Args:
            texts: Original text strings
            current_embeddings: Current student embeddings [N, dim]
            teacher_weight: Weight for teacher guidance (0=student only, 1=teacher only)
            
        Returns:
            Target embeddings [N, dim]
        """
        if teacher_weight == 0.0:
            # Pure student mode - no teacher guidance
            return current_embeddings.copy()
        
        # Get teacher embeddings
        teacher_embeds = self.get_teacher_embeddings(texts)
        
        if len(teacher_embeds) == 0 or len(teacher_embeds) != len(current_embeddings):
            self.logger.log_event(
                "distillation_target_fallback",
                "Teacher embedding mismatch, using student embeddings only",
                level="WARNING",
                teacher_size=len(teacher_embeds),
                student_size=len(current_embeddings)
            )
            return current_embeddings.copy()
        
        # Check dimension compatibility
        if teacher_embeds.shape[1] != current_embeddings.shape[1]:
            self.logger.log_event(
                "dimension_mismatch_targets",
                f"Cannot blend: teacher dim {teacher_embeds.shape[1]} != student dim {current_embeddings.shape[1]}",
                level="ERROR",
                teacher_dim=teacher_embeds.shape[1],
                student_dim=current_embeddings.shape[1]
            )
            return current_embeddings.copy()
        
        # Blend: target = (1 - alpha) * student + alpha * teacher
        alpha = teacher_weight
        blended = (1 - alpha) * current_embeddings + alpha * teacher_embeds
        
        # Normalize to unit sphere (important for cosine similarity)
        norms = np.linalg.norm(blended, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)  # Avoid division by zero
        targets = blended / norms
        
        self.logger.log_event(
            "distillation_targets_generated",
            f"Generated {len(targets)} distillation targets",
            level="DEBUG",
            num_targets=len(targets),
            teacher_weight=teacher_weight
        )
        
        return targets
    
    def compute_alignment_metric(
        self,
        student_embeds: np.ndarray,
        teacher_embeds: np.ndarray
    ) -> float:
        """
        Compute average cosine similarity between student and teacher embeddings.
        
        Args:
            student_embeds: Student embeddings [N, dim]
            teacher_embeds: Teacher embeddings [N, dim]
            
        Returns:
            Average cosine similarity (0-1)
        """
        if len(student_embeds) == 0 or len(teacher_embeds) == 0:
            return 0.0
        
        if student_embeds.shape != teacher_embeds.shape:
            self.logger.log_event(
                "alignment_shape_mismatch",
                "Shape mismatch in alignment computation",
                level="WARNING",
                student_shape=student_embeds.shape,
                teacher_shape=teacher_embeds.shape
            )
            return 0.0
        
        # Compute row-wise cosine similarities
        student_norm = student_embeds / (np.linalg.norm(student_embeds, axis=1, keepdims=True) + 1e-9)
        teacher_norm = teacher_embeds / (np.linalg.norm(teacher_embeds, axis=1, keepdims=True) + 1e-9)
        
        similarities = np.sum(student_norm * teacher_norm, axis=1)
        avg_similarity = np.mean(similarities)
        
        return float(avg_similarity)


def create_distillation_helper(
    teacher_model_name: Optional[str] = None
) -> TeacherDistillationHelper:
    """
    Factory function to create distillation helper.
    
    Args:
        teacher_model_name: Optional teacher model name (uses default if None)
        
    Returns:
        TeacherDistillationHelper instance
    """
    if teacher_model_name is None:
        teacher_model_name = ChelationConfig.DEFAULT_TEACHER_MODEL
    
    return TeacherDistillationHelper(teacher_model_name=teacher_model_name)


# Convenience function for hybrid target generation
def generate_hybrid_targets(
    texts: List[str],
    current_embeddings: np.ndarray,
    homeostatic_targets: np.ndarray,
    teacher_weight: float = 0.5,
    teacher_helper: Optional[TeacherDistillationHelper] = None
) -> np.ndarray:
    """
    Generate hybrid targets by blending homeostatic and teacher guidance.
    
    Args:
        texts: Original text strings
        current_embeddings: Current embeddings [N, dim]
        homeostatic_targets: Homeostatic push targets [N, dim]
        teacher_weight: Weight for teacher vs homeostatic (0=homeostatic only, 1=teacher only)
        teacher_helper: Optional pre-initialized helper (creates new if None)
        
    Returns:
        Hybrid target embeddings [N, dim]
    """
    if teacher_helper is None:
        teacher_helper = create_distillation_helper()
    
    if teacher_weight == 0.0:
        # Pure homeostatic mode
        return homeostatic_targets.copy()
    
    # Get teacher embeddings
    teacher_embeds = teacher_helper.get_teacher_embeddings(texts)
    
    if len(teacher_embeds) == 0 or teacher_embeds.shape != current_embeddings.shape:
        # Fallback to homeostatic only
        return homeostatic_targets.copy()
    
    # Blend: target = (1 - alpha) * homeostatic + alpha * teacher
    alpha = teacher_weight
    blended = (1 - alpha) * homeostatic_targets + alpha * teacher_embeds
    
    # Normalize
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    targets = blended / norms
    
    return targets
