"""Versioned index, ledger, and stream profiles plus capability negotiation."""

from __future__ import annotations

from dataclasses import dataclass

from chelated_kernel.errors import ProfileError, UnsupportedCapabilityError, TenantIsolationError

INDEX_PROFILE_VERSION = "1.0.0"
LEDGER_POLICY_VERSION = "1.0.0"
STREAM_PROFILE_VERSION = "1.0.0"
KERNEL_VERSION = "1.0.0"

_TENANT_RE_DESC = "tenant_id must match [A-Za-z0-9._:-]{1,64}"


@dataclass(frozen=True)
class IndexProfile:
    name: str
    version: str
    bridge: str
    scorer: str
    sidecar_bytes_per_doc: int
    min_fit_count: int
    min_validation_count: int
    minimum_route_confidence: float
    csls_k: int
    bootstrap_resamples: int
    bootstrap_alpha: float


@dataclass(frozen=True)
class LedgerPolicy:
    name: str
    version: str
    code: int
    default_authority: int
    redact_on_erase: bool


@dataclass(frozen=True)
class StreamProfile:
    name: str
    version: str
    code: int
    mode: str


@dataclass(frozen=True)
class CapabilityGrant:
    """Typed grant returned by negotiate_capability."""

    name: str
    version: str
    tenant_scope: str
    parameters: tuple[tuple[str, str], ...]


INDEX_PROFILES: dict[str, IndexProfile] = {
    "enterprise_search": IndexProfile(
        name="enterprise_search",
        version=INDEX_PROFILE_VERSION,
        bridge="zeromu",
        scorer="csls",
        sidecar_bytes_per_doc=8,
        min_fit_count=32,
        min_validation_count=16,
        minimum_route_confidence=0.70,
        csls_k=10,
        bootstrap_resamples=1000,
        bootstrap_alpha=0.05,
    ),
}

LEDGER_POLICIES: dict[str, LedgerPolicy] = {
    "sec_audit": LedgerPolicy("sec_audit", LEDGER_POLICY_VERSION, 1, 80, False),
    "cve_vex": LedgerPolicy("cve_vex", LEDGER_POLICY_VERSION, 2, 70, False),
    "gdpr_erasure": LedgerPolicy("gdpr_erasure", LEDGER_POLICY_VERSION, 3, 90, True),
    "dpu_fabric": LedgerPolicy("dpu_fabric", LEDGER_POLICY_VERSION, 4, 60, False),
    "agent_pool": LedgerPolicy("agent_pool", LEDGER_POLICY_VERSION, 5, 50, False),
}

STREAM_PROFILES: dict[str, StreamProfile] = {
    "strict_holdback": StreamProfile("strict_holdback", STREAM_PROFILE_VERSION, 1, "strict_holdback"),
    "async_armed": StreamProfile("async_armed", STREAM_PROFILE_VERSION, 2, "async_armed"),
}

# Names the wheel actually implements. Unknown names fail closed.
SUPPORTED_CAPABILITIES: frozenset[str] = frozenset(
    {
        "wrap_index",
        "event_ledger",
        "wrap_stream",
        "tra1_verify",
        "tenant_partition",
        "ed25519_receipt",
        "as_of_query",
        "csls_sidecar",
        "zeromu_bridge",
        "cloudevents_1_0",
        "w3c_prov",
        "osv_csaf_vex",
        "git_commit_symbol",
        "cdc_envelope",
        "echo_collapse",
        "cites_audit",
        "gdpr_erasure_tombstone",
        "non_nda_receipt",
        "enterprise_search",
        "strict_holdback",
        "async_armed",
    }
)

# Planted out-of-scope canaries. These always raise UnsupportedCapabilityError.
UNSUPPORTED_CANARIES: frozenset[str] = frozenset(
    {
        "bare_metal_p4_tcam_asic_dataplane",
        "physical_weight_erasure_unlearning",
        "hdd_servo_mcu_firmware",
        "cross_tenant_raw_join",
    }
)

_CROSS_TENANT_KEYS = frozenset(
    {
        "cross_tenant",
        "cross_tenant_raw_join",
        "tenant_join",
        "foreign_tenant",
        "override_tenant",
    }
)


def require_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not tenant_id:
        raise TenantIsolationError(_TENANT_RE_DESC)
    if len(tenant_id) > 64:
        raise TenantIsolationError(_TENANT_RE_DESC)
    for char in tenant_id:
        ok = ("A" <= char <= "Z") or ("a" <= char <= "z") or ("0" <= char <= "9") or char in "._:-"
        if not ok:
            raise TenantIsolationError(_TENANT_RE_DESC)
    return tenant_id


def get_index_profile(name: str) -> IndexProfile:
    try:
        return INDEX_PROFILES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown index profile {name!r}") from exc


def get_ledger_policy(name: str) -> LedgerPolicy:
    try:
        return LEDGER_POLICIES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown ledger policy {name!r}") from exc


def get_stream_profile(name: str) -> StreamProfile:
    try:
        return STREAM_PROFILES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown stream profile {name!r}") from exc


def negotiate_capability(capability_name: str, **kwargs: object) -> CapabilityGrant:
    """Return a typed grant, or raise UnsupportedCapabilityError.

    Unknown names fail closed. The four planted canaries always fail.
    A supported name that asks to cross tenants also fails.
    """

    if not isinstance(capability_name, str) or not capability_name:
        raise UnsupportedCapabilityError(str(capability_name), reason="capability name must be a non-empty string")
    for key in kwargs:
        if key in _CROSS_TENANT_KEYS or key == "tenant_id" and kwargs[key] == "*":
            raise UnsupportedCapabilityError(capability_name, reason="cross-tenant raw access is outside the wheel")
        if kwargs[key] is True and "cross_tenant" in key:
            raise UnsupportedCapabilityError(capability_name, reason="cross-tenant raw access is outside the wheel")
    if capability_name in UNSUPPORTED_CANARIES or capability_name not in SUPPORTED_CAPABILITIES:
        raise UnsupportedCapabilityError(
            capability_name,
            reason="not implemented in chelated-kernel 1.0.0",
        )
    tenant_scope = kwargs.get("tenant_id", "caller")
    if not isinstance(tenant_scope, str):
        raise UnsupportedCapabilityError(capability_name, reason="tenant_id must be a string when provided")
    if tenant_scope != "caller":
        require_tenant_id(tenant_scope)
    items: list[tuple[str, str]] = []
    for key in sorted(kwargs):
        if key == "tenant_id":
            continue
        value = kwargs[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            items.append((key, repr(value)))
        else:
            raise UnsupportedCapabilityError(capability_name, reason=f"parameter {key} is not a scalar")
    return CapabilityGrant(
        name=capability_name,
        version=KERNEL_VERSION,
        tenant_scope=tenant_scope,
        parameters=tuple(items),
    )


def probe_import_barrier(module_name: str) -> None:
    """Import a module from inside this package so a gate finder can fail closed.

    The cold-start gate installs a meta-path finder and calls this probe. Package
    code has no static import of a repository module; the name arrives as an argument.
    """

    if not isinstance(module_name, str) or not module_name:
        raise ProfileError("barrier probe requires a module name")
    if module_name == "chelated_kernel" or module_name.startswith("chelated_kernel."):
        raise ProfileError("barrier probe does not target the kernel package")
    __import__(module_name)
