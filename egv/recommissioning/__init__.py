"""Evidence-bounded Nemotron recommissioning components."""

from importlib import import_module

from .client import (
    AttestedLoopbackService,
    BoundedModelClient,
    CoordinateTokenDebit,
    CoordinateTokenLedger,
    CoordinateTokenState,
    FileCoordinateTokenLedger,
    GenerationReceipt,
    GenerationRequest,
    GenerationResult,
    InMemoryCoordinateTokenLedger,
    ModelServiceProfile,
    PinnedTokenizerRenderer,
    PrivateGenerationRecord,
    PublicGenerationReceipt,
    RecommissioningClientError,
    ServiceAttestation,
    TokenizationPreflight,
    TrustedLaunchReceipt,
    TypedCandidateResponse,
    UrllibLoopbackTransport,
    candidate_response_schema,
)
from .protocol import (
    AttemptFeedback,
    ChatMessage,
    DIAGNOSTIC_IDS,
    GenerationBudget,
    MODEL_BUNDLE_IDS,
    PROTOCOL_IDS,
    PilotCoordinate,
    PromptEnvelope,
    RecommissioningProtocolError,
    SignedEvaluatorReceipt,
    TokenBudget,
    VisibleTaskPackage,
    lineage_feedback,
    public_task_ref,
    render_prompt,
)

__all__ = [
    "AttestedLoopbackService",
    "AttemptFeedback",
    "BoundedModelClient",
    "ChatMessage",
    "CoordinateTokenDebit",
    "CoordinateTokenLedger",
    "CoordinateTokenState",
    "DIAGNOSTIC_IDS",
    "FileCoordinateTokenLedger",
    "GenerationBudget",
    "GenerationReceipt",
    "GenerationRequest",
    "GenerationResult",
    "GatewayConfig",
    "InMemoryCoordinateTokenLedger",
    "LoopbackAttestationGateway",
    "MODEL_BUNDLE_IDS",
    "ModelServiceProfile",
    "PinnedTokenizerRenderer",
    "PROTOCOL_IDS",
    "PilotCoordinate",
    "PrivateGenerationRecord",
    "PromptEnvelope",
    "PublicGenerationReceipt",
    "RecommissioningClientError",
    "RecommissioningGatewayError",
    "RecommissioningProtocolError",
    "ServiceAttestation",
    "SignedEvaluatorReceipt",
    "TokenBudget",
    "TokenizationPreflight",
    "TrustedLaunchReceipt",
    "TypedCandidateResponse",
    "UrllibLoopbackTransport",
    "VisibleTaskPackage",
    "build_gateway_server",
    "candidate_response_schema",
    "issue_trusted_launch_receipt",
    "lineage_feedback",
    "public_task_ref",
    "render_prompt",
    "sign_service_observation",
]


_GATEWAY_EXPORTS = {
    "GatewayConfig",
    "LoopbackAttestationGateway",
    "RecommissioningGatewayError",
    "build_gateway_server",
    "issue_trusted_launch_receipt",
    "sign_service_observation",
}


def __getattr__(name):  # type: ignore[no-untyped-def]
    if name in _GATEWAY_EXPORTS:
        return getattr(import_module(".gateway", __name__), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
