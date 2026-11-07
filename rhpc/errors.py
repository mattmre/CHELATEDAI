"""Closed error vocabulary for RHPC method development."""


class RHPCError(RuntimeError):
    """Base class for RHPC contract failures."""


class RHPCConfigurationError(RHPCError):
    """The frozen experiment configuration is malformed."""


class RHPCResultError(RHPCError):
    """A result or artifact does not satisfy its closed contract."""


class RHPCAdmissionError(RHPCError):
    """The predecessor campaign or Spark execution boundary is not satisfied."""


__all__ = [
    "RHPCAdmissionError",
    "RHPCConfigurationError",
    "RHPCError",
    "RHPCResultError",
]
