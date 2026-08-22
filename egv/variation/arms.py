"""Frozen arm definitions and filesystem isolation for Variation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..canonical import digest_for
from .errors import VariationConfigurationError, VariationIsolationError


ARM_IDS = ("A", "B", "C", "D", "E", "F", "G", "H")
MODEL_BASE = "BASE"
MODEL_LORA = "LORA"


@dataclass(frozen=True)
class ArmPolicy:
    arm_id: str
    model_mode: str
    retrieval_policy: str
    authority_enforced: bool
    requires_adapter: bool

    def validate(self) -> None:
        if self.arm_id not in ARM_IDS:
            raise VariationConfigurationError("unknown frozen Variation arm")
        if self.model_mode not in {MODEL_BASE, MODEL_LORA}:
            raise VariationConfigurationError("arm model mode is outside the frozen contract")
        if self.requires_adapter != (self.model_mode == MODEL_LORA):
            raise VariationConfigurationError("arm adapter requirement does not match its model mode")
        if not isinstance(self.authority_enforced, bool):
            raise VariationConfigurationError("arm authority policy must be boolean")

    @property
    def digest(self) -> str:
        self.validate()
        return digest_for(
            {
                "arm_id": self.arm_id,
                "model_mode": self.model_mode,
                "retrieval_policy": self.retrieval_policy,
                "authority_enforced": self.authority_enforced,
                "requires_adapter": self.requires_adapter,
            }
        )


_POLICIES = {
    "A": ArmPolicy("A", MODEL_BASE, "SUCCESS_ONLY", False, False),
    "B": ArmPolicy("B", MODEL_BASE, "ORDINARY_FAILURE_SUMMARY", False, False),
    "C": ArmPolicy("C", MODEL_BASE, "CORRECTION_AWARE", False, False),
    "D": ArmPolicy("D", MODEL_BASE, "CORRECTION_AWARE", True, False),
    "E": ArmPolicy("E", MODEL_LORA, "CORRECTION_AWARE", True, True),
    "F": ArmPolicy("F", MODEL_LORA, "SUCCESS_ONLY", False, True),
    "G": ArmPolicy("G", MODEL_LORA, "ORDINARY_FAILURE_SUMMARY", False, True),
    "H": ArmPolicy("H", MODEL_LORA, "CORRECTION_AWARE", False, True),
}


def arm_policy(arm_id: str) -> ArmPolicy:
    try:
        policy = _POLICIES[arm_id]
    except KeyError as exc:
        raise VariationConfigurationError("unknown frozen Variation arm: {}".format(arm_id)) from exc
    policy.validate()
    return policy


@dataclass(frozen=True)
class ArmWorkspace:
    arm_id: str
    run_id: str
    root: Path
    candidates: Path
    checkpoints: Path
    retrieval: Path


class ArmIsolation:
    """Create and verify disjoint per-arm/per-run state roots."""

    def __init__(self, root: Path, *, campaign_id: str) -> None:
        self.root = Path(root).resolve()
        self.campaign_id = campaign_id
        if not campaign_id or "/" in campaign_id or "\\" in campaign_id:
            raise VariationConfigurationError("campaign ID is not a bounded path component")
        self.root.mkdir(parents=True, exist_ok=True)
        self._workspaces: Dict[Tuple[str, str], ArmWorkspace] = {}
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": "egv-variation-arm-isolation-v1",
            "campaign_id": self.campaign_id,
            "arm_ids": list(ARM_IDS),
            "arm_policy_digests": {arm_id: arm_policy(arm_id).digest for arm_id in ARM_IDS},
        }
        path = self.root / "arm-isolation-manifest.json"
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise VariationIsolationError("arm isolation manifest is unreadable") from exc
            if existing != manifest:
                raise VariationIsolationError("arm isolation manifest conflicts with the frozen arm contract")
            return
        path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        path.chmod(0o444)

    @staticmethod
    def _safe_component(value: str, label: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise VariationIsolationError("{} is not a bounded path component".format(label))
        return value

    def workspace(self, arm_id: str, run_id: str) -> ArmWorkspace:
        policy = arm_policy(arm_id)
        self._safe_component(run_id, "run ID")
        key = (policy.arm_id, run_id)
        existing = self._workspaces.get(key)
        if existing is not None:
            return existing
        arm_root = (self.root / "arms" / policy.arm_id / "runs" / run_id).resolve()
        try:
            arm_root.relative_to(self.root)
        except ValueError as exc:
            raise VariationIsolationError("arm workspace escapes isolation root") from exc
        arm_root.mkdir(parents=True, exist_ok=True)
        workspace = ArmWorkspace(
            policy.arm_id,
            run_id,
            arm_root,
            arm_root / "candidates",
            arm_root / "checkpoints",
            arm_root / "retrieval",
        )
        for path in (workspace.candidates, workspace.checkpoints, workspace.retrieval):
            path.mkdir(parents=True, exist_ok=True)
        self._workspaces[key] = workspace
        self.assert_disjoint()
        return workspace

    def assert_disjoint(self) -> None:
        roots = [workspace.root.resolve() for workspace in self._workspaces.values()]
        if len(roots) != len(set(roots)):
            raise VariationIsolationError("two Variation arm/run contexts share a workspace")
        for index, left in enumerate(roots):
            for right in roots[index + 1 :]:
                if left in right.parents or right in left.parents:
                    raise VariationIsolationError("Variation arm/run workspaces overlap")

    def assert_candidate_id(self, arm_id: str, candidate_id: str) -> None:
        prefix = "egv-candidate-{}-{}-".format(self.campaign_id, arm_id)
        if not candidate_id.startswith(prefix):
            raise VariationIsolationError("candidate ID is not bound to its Variation arm")

    def assert_retrieval_arm(self, expected_arm: str, event_arm: Optional[str]) -> None:
        if event_arm != expected_arm:
            raise VariationIsolationError("cross-arm evidence retrieval was attempted")


__all__ = [
    "ARM_IDS",
    "ArmIsolation",
    "ArmPolicy",
    "ArmWorkspace",
    "MODEL_BASE",
    "MODEL_LORA",
    "arm_policy",
]
