"""Evaluation-slice error taxonomy."""

from ..errors import EGVError


class EvaluationError(EGVError):
    """Base class for bounded Evaluation-slice failures."""


class ArtifactError(EvaluationError):
    """Raised when a content-addressed or frozen artifact is invalid."""


class LeakageError(EvaluationError):
    """Raised when split or private-material leakage is detected."""


class AuthorityDenied(EvaluationError):
    """Raised when deny-by-default authority does not allow an action."""


class InfrastructureFailure(EvaluationError):
    """Raised for evaluator infrastructure loss, never for model failure."""


class DockerConfigurationError(InfrastructureFailure):
    """The configured production isolation runtime is absent or unpinned."""


class ShockMismatchError(EvaluationError):
    """Raised when a correction-shock clone is not an exact matched clone."""


__all__ = [
    "ArtifactError",
    "AuthorityDenied",
    "DockerConfigurationError",
    "EvaluationError",
    "InfrastructureFailure",
    "LeakageError",
    "ShockMismatchError",
]
