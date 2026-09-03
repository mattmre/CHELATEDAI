"""Fail-closed errors for the EGV Campaign control plane."""


class CampaignError(RuntimeError):
    """Base campaign control-plane failure."""


class CampaignStateError(CampaignError):
    """The durable campaign lifecycle state is invalid or unsafe."""


class ProtectedInventoryError(CampaignError):
    """The protected restore inventory is ambiguous or unsafe."""


class LifecycleExecutionError(CampaignError):
    """A planned service lifecycle action failed verification."""


__all__ = [
    "CampaignError",
    "CampaignStateError",
    "LifecycleExecutionError",
    "ProtectedInventoryError",
]
