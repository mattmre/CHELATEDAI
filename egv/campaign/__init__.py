"""Fail-closed orchestration primitives for the EGV Spark campaign."""

from .authority import EvaluatorAuthority, EvaluatorReceipt
from .commissioning import CommissioningPlan, prepare_commissioning
from .coordinator import CampaignCoordinator
from .inventory import ProtectedRestoreInventory
from .lifecycle import DeepSeekLifecycleController
from .state import CampaignState, CampaignStateStore
from .transport import ArtifactStagingStore, TransferManifest
from .trajectories import GenerationRequest, GenerationResponse, reconcile_responses
from .runner import (
    CommissioningTrainerInputs,
    CommissioningTrainerSources,
    freeze_commissioning_dataset,
    run_commissioning,
)

__all__ = [
    "ArtifactStagingStore",
    "CampaignCoordinator",
    "CommissioningPlan",
    "CommissioningTrainerInputs",
    "CommissioningTrainerSources",
    "CampaignState",
    "CampaignStateStore",
    "DeepSeekLifecycleController",
    "EvaluatorAuthority",
    "EvaluatorReceipt",
    "GenerationRequest",
    "GenerationResponse",
    "ProtectedRestoreInventory",
    "TransferManifest",
    "prepare_commissioning",
    "freeze_commissioning_dataset",
    "reconcile_responses",
    "run_commissioning",
]
