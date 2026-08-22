"""Shared content-derived identities used across orchestration layers."""

from __future__ import annotations

from .canonical import digest_for


def commissioning_run_id(*, campaign_id: str, task_id: str, arm_id: str, seed: int) -> str:
    """Return the sole run identity for one frozen commissioning coordinate."""

    if any(not isinstance(value, str) or not value or "/" in value or "\\" in value
           for value in (campaign_id, task_id, arm_id)):
        raise ValueError("commissioning run identity contains an invalid component")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("commissioning run seed must be nonnegative")
    return "egv-run-{}".format(
        digest_for({"campaign_id": campaign_id, "task_id": task_id, "arm_id": arm_id, "seed": seed})[:40]
    )


__all__ = ["commissioning_run_id"]
