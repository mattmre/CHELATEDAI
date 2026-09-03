"""Fail-closed errors for the EGV Training slice."""


class TrainingArtifactError(RuntimeError):
    """A training checkpoint or artifact failed integrity validation."""


class TrainingCheckpointError(TrainingArtifactError):
    """A resumable training checkpoint is malformed, stale, or unsafe."""


__all__ = ["TrainingArtifactError", "TrainingCheckpointError"]
