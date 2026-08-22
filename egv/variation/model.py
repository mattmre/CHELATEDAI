"""Revision-pinned, local-only Qwen model loading for the Variation slice.

The loader deliberately does not download, resolve a mutable branch, enable
remote code, or fall back to a different model class.  A model directory must
carry a closed manifest with file hashes before Transformers is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import digest_for
from .errors import VariationConfigurationError, VariationDependencyError


MODEL_REPOSITORY = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
MODEL_ARCHITECTURE = "Qwen3_5ForCausalLM"
MODEL_CONFIG_CLASS = "Qwen3_5TextConfig"
TRANSFORMERS_MIN_VERSION = (5, 5, 0)
TRANSFORMERS_MAX_EXCLUSIVE = (6, 0, 0)
MODEL_MANIFEST_SCHEMA = "egv-pinned-model-v1"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "repository",
        "revision",
        "architecture",
        "config_class",
        "transformers_version",
        "files",
        "license",
    }
)
_ALLOWED_UNUSED_KEY_PREFIXES = ("visual.", "vision_", "mtp.", "mtp_")


def _version_tuple(value: str) -> Tuple[int, int, int]:
    parts = value.split(".")
    numbers = []
    for part in parts[:3]:
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            raise VariationDependencyError("Transformers version is not numeric: {}".format(value))
        numbers.append(int(digits))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


def _relative_file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise VariationConfigurationError("model manifest contains an absolute or empty file path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise VariationConfigurationError("model manifest file escapes the model root") from exc
    return candidate


@dataclass(frozen=True)
class PinnedModelManifest:
    """Closed manifest required before a checkpoint can be loaded."""

    repository: str
    revision: str
    architecture: str
    config_class: str
    transformers_version: str
    files: Mapping[str, str]
    license: Mapping[str, str]
    schema_version: str = MODEL_MANIFEST_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedModelManifest":
        if set(value) != _MANIFEST_FIELDS:
            raise VariationConfigurationError("pinned model manifest has an unexpected field set")
        if value.get("schema_version") != MODEL_MANIFEST_SCHEMA:
            raise VariationConfigurationError("unsupported pinned model manifest schema")
        files = value.get("files")
        license_metadata = value.get("license")
        if not isinstance(files, Mapping) or not files:
            raise VariationConfigurationError("pinned model manifest must hash at least one file")
        if not isinstance(license_metadata, Mapping) or not license_metadata:
            raise VariationConfigurationError("pinned model manifest must record license metadata")
        normalized_files: Dict[str, str] = {}
        for path, digest in files.items():
            if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
                raise VariationConfigurationError("model file manifest entries must be path/SHA-256 pairs")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise VariationConfigurationError("model file manifest contains a non-hex digest") from exc
            normalized_files[path] = digest
        normalized_license = {str(key): str(item) for key, item in license_metadata.items()}
        result = cls(
            repository=str(value["repository"]),
            revision=str(value["revision"]),
            architecture=str(value["architecture"]),
            config_class=str(value["config_class"]),
            transformers_version=str(value["transformers_version"]),
            files=normalized_files,
            license=normalized_license,
            schema_version=str(value["schema_version"]),
        )
        result.validate_contract()
        return result

    @classmethod
    def from_file(cls, path: Path) -> "PinnedModelManifest":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VariationConfigurationError("pinned model manifest cannot be read") from exc
        if not isinstance(value, Mapping):
            raise VariationConfigurationError("pinned model manifest must be a JSON object")
        return cls.from_mapping(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "revision": self.revision,
            "architecture": self.architecture,
            "config_class": self.config_class,
            "transformers_version": self.transformers_version,
            "files": dict(sorted(self.files.items())),
            "license": dict(sorted(self.license.items())),
        }

    def digest(self) -> str:
        return digest_for(self.to_dict())

    def validate_contract(self) -> None:
        if self.repository != MODEL_REPOSITORY:
            raise VariationConfigurationError("model repository is not the frozen Qwen checkpoint")
        if self.revision != MODEL_REVISION:
            raise VariationConfigurationError("model revision is not the frozen immutable revision")
        if self.architecture != MODEL_ARCHITECTURE or self.config_class != MODEL_CONFIG_CLASS:
            raise VariationConfigurationError("model architecture is not the frozen text-only contract")
        version = _version_tuple(self.transformers_version)
        if version < TRANSFORMERS_MIN_VERSION or version >= TRANSFORMERS_MAX_EXCLUSIVE:
            raise VariationConfigurationError("model manifest Transformers version is outside >=5.5,<6")
        if not self.license.get("name") or not self.license.get("source"):
            raise VariationConfigurationError("model license metadata must include name and source")


@dataclass(frozen=True)
class LoadedPinnedModel:
    """Loaded text-only model plus the immutable proof captured at load time."""

    model: Any
    tokenizer: Any
    manifest: PinnedModelManifest
    manifest_digest: str
    file_hashes: Mapping[str, str]
    load_report: Mapping[str, Any]
    base_state_digest: Optional[str]
    adapter_digest: Optional[str] = None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model: Any) -> Optional[str]:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        return None
    records = []
    for name, tensor in sorted(state_dict().items()):
        try:
            material = tensor.detach().cpu().numpy().tobytes()
            shape = tuple(int(value) for value in tensor.shape)
            dtype = str(tensor.dtype)
        except AttributeError:
            material = bytes(tensor)
            shape = ()
            dtype = type(tensor).__name__
        records.append({"name": name, "shape": shape, "dtype": dtype, "digest": hashlib.sha256(material).hexdigest()})
    return digest_for(records)


class PinnedModelLoader:
    """Load only a locally staged, revision-verified Qwen text checkpoint."""

    def __init__(self, model_root: Path, *, manifest_path: Optional[Path] = None) -> None:
        self.model_root = Path(model_root)
        self.manifest_path = manifest_path or self.model_root / "model-manifest.json"

    def verify_manifest(self) -> Tuple[PinnedModelManifest, Dict[str, str]]:
        root = self.model_root.resolve()
        if not root.is_dir():
            raise VariationConfigurationError("pinned model root does not exist or is not a directory")
        manifest_path = self.manifest_path.resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise VariationConfigurationError("pinned model manifest is outside the model root") from exc
        manifest = PinnedModelManifest.from_file(manifest_path)
        expected_paths = set(manifest.files)
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest_path
        }
        if actual_paths != expected_paths:
            extra = sorted(actual_paths - expected_paths)
            missing = sorted(expected_paths - actual_paths)
            raise VariationConfigurationError(
                "pinned model tree is not exhaustive (extra={}, missing={})".format(extra, missing)
            )
        file_hashes: Dict[str, str] = {}
        for relative_path, expected_digest in sorted(manifest.files.items()):
            path = _relative_file(root, relative_path)
            if not path.is_file() or path.is_symlink():
                raise VariationConfigurationError("pinned model file is missing: {}".format(relative_path))
            actual_digest = _hash_file(path)
            if actual_digest != expected_digest:
                raise VariationConfigurationError("pinned model file digest mismatch: {}".format(relative_path))
            file_hashes[relative_path] = actual_digest
        return manifest, file_hashes

    def preflight(self) -> Dict[str, Any]:
        manifest, file_hashes = self.verify_manifest()
        return {
            "repository": manifest.repository,
            "revision": manifest.revision,
            "architecture": manifest.architecture,
            "config_class": manifest.config_class,
            "manifest_digest": manifest.digest(),
            "file_count": len(file_hashes),
            "file_hashes": file_hashes,
            "license": dict(manifest.license),
            "network": "disabled-local-files-only",
        }

    @staticmethod
    def _transformers_version() -> str:
        try:
            return importlib_metadata.version("transformers")
        except importlib_metadata.PackageNotFoundError as exc:
            raise VariationDependencyError("transformers>=5.5,<6 is required for the pinned model") from exc

    @staticmethod
    @contextmanager
    def _offline_environment() -> Any:
        """Force the Hugging Face stack offline for the complete load window."""

        names = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        previous = {name: os.environ.get(name) for name in names}
        try:
            os.environ.update(names)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def load(
        self,
        *,
        device: str = "cpu",
        torch_dtype: Optional[Any] = None,
        adapter_artifact: Optional[Any] = None,
    ) -> LoadedPinnedModel:
        manifest, file_hashes = self.verify_manifest()
        installed_version = self._transformers_version()
        if not (TRANSFORMERS_MIN_VERSION <= _version_tuple(installed_version) < TRANSFORMERS_MAX_EXCLUSIVE):
            raise VariationDependencyError("installed Transformers is outside the frozen >=5.5,<6 contract")
        if adapter_artifact is not None:
            from .adapter import SealedAdapterArtifact

            if not isinstance(adapter_artifact, SealedAdapterArtifact):
                raise VariationDependencyError("model adapters must be sealed content-addressed artifacts")
        with self._offline_environment():
            try:
                transformers = importlib.import_module("transformers")
                causal_class = getattr(transformers, MODEL_ARCHITECTURE)
                config_class = getattr(transformers, MODEL_CONFIG_CLASS)
                tokenizer_class = getattr(transformers, "AutoTokenizer")
            except (ImportError, AttributeError) as exc:
                raise VariationDependencyError("installed Transformers lacks the frozen Qwen text-only classes") from exc
            try:
                config = config_class.from_pretrained(
                    str(self.model_root), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
                )
                if type(config).__name__ != MODEL_CONFIG_CLASS:
                    raise VariationConfigurationError("checkpoint config is not Qwen3_5TextConfig")
                load_kwargs: Dict[str, Any] = {
                    "config": config,
                    "revision": MODEL_REVISION,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "output_loading_info": True,
                }
                if torch_dtype is not None:
                    load_kwargs["torch_dtype"] = torch_dtype
                loaded = causal_class.from_pretrained(str(self.model_root), **load_kwargs)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    model, loading_info = loaded
                else:
                    model, loading_info = loaded, {}
                model_config = getattr(model, "config", None)
                if type(model_config).__name__ != MODEL_CONFIG_CLASS:
                    raise VariationConfigurationError("loaded model does not expose Qwen3_5TextConfig")
                tokenizer = tokenizer_class.from_pretrained(
                    str(self.model_root), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
                )
                base_state_digest = _state_digest(model)
                adapter_digest = None
                if adapter_artifact is not None:
                    adapter_digest = adapter_artifact.digest
                    adapter_artifact.verify()
                    model = adapter_artifact.apply_to(model)
            except VariationConfigurationError:
                raise
            except Exception as exc:
                raise VariationDependencyError("pinned text-only checkpoint failed local loading") from exc
        report = dict(loading_info) if isinstance(loading_info, Mapping) else {}
        missing = list(report.get("missing_keys", ()))
        unexpected = list(report.get("unexpected_keys", ()))
        if missing or any(
            not any(str(key).startswith(prefix) for prefix in _ALLOWED_UNUSED_KEY_PREFIXES) for key in unexpected
        ):
            raise VariationConfigurationError(
                "checkpoint tensor load report is outside the frozen language-model/visual-MTP contract"
            )
        if device != "cpu":
            try:
                model = model.to(device)
            except Exception as exc:
                raise VariationDependencyError("pinned model could not move to the requested device") from exc
        return LoadedPinnedModel(
            model=model,
            tokenizer=tokenizer,
            manifest=manifest,
            manifest_digest=manifest.digest(),
            file_hashes=file_hashes,
            load_report={"missing_keys": missing, "unexpected_keys": unexpected, "transformers_version": installed_version},
            base_state_digest=base_state_digest,
            adapter_digest=adapter_digest,
        )


def build_local_manifest(
    model_root: Path,
    *,
    transformers_version: str = "5.5.0",
    license_name: str = "unknown",
    license_source: str = "local-staging-record",
) -> PinnedModelManifest:
    """Build a manifest for an already staged local checkpoint directory.

    This helper hashes bytes only; it does not fetch or validate a model.  The
    operator must supply truthful license metadata before using the manifest.
    """

    files: Dict[str, str] = {}
    manifest_path = (model_root / "model-manifest.json").resolve()
    for path in sorted(model_root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.resolve() != manifest_path:
            files[str(path.relative_to(model_root))] = _hash_file(path)
    return PinnedModelManifest.from_mapping(
        {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "architecture": MODEL_ARCHITECTURE,
            "config_class": MODEL_CONFIG_CLASS,
            "transformers_version": transformers_version,
            "files": files,
            "license": {"name": license_name, "source": license_source},
        }
    )


__all__ = [
    "LoadedPinnedModel",
    "MODEL_ARCHITECTURE",
    "MODEL_CONFIG_CLASS",
    "MODEL_MANIFEST_SCHEMA",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "PinnedModelLoader",
    "PinnedModelManifest",
    "TRANSFORMERS_MAX_EXCLUSIVE",
    "TRANSFORMERS_MIN_VERSION",
    "build_local_manifest",
]
