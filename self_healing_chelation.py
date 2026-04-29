"""SEAL-inspired self-edit planning for self-healing chelation workflows.

The module translates runtime diagnostics and new context into candidate
self-edits, then filters them with outcome fitness, retention checks, and
quantization survival. It is advisory by default: accepted directives describe
what should be adapted, but they do not mutate the base embedding model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from chelation_logger import get_logger
from fitness_interfaces import FitnessEvaluation, FitnessFunctionInterface
from quantization_promotion_gate import QuantizationPromotionGate


@dataclass
class SelfEditDirective:
    """A candidate update directive generated from context and diagnostics."""

    directive_id: str
    strategy: str
    adaptation_mode: str
    synthetic_examples: List[str] = field(default_factory=list)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    source: str = "seal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfEditEvaluation:
    """Outcome record for one self-edit candidate."""

    directive: SelfEditDirective
    fitness: FitnessEvaluation
    reward: float
    accepted: bool
    reasons: List[str] = field(default_factory=list)
    quantization_gate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directive": self.directive.to_dict(),
            "fitness": {
                "candidate_id": self.fitness.candidate_id,
                "fitness": self.fitness.fitness,
                "metrics": self.fitness.metrics,
                "metadata": self.fitness.metadata,
            },
            "reward": self.reward,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "quantization_gate": self.quantization_gate,
        }


@dataclass
class SelfGeneratedEvalProbe:
    """Synthetic probe used to test whether a self-edit preserves useful retrieval facts."""

    probe_id: str
    question: str
    expected_terms: List[str]
    probe_type: str = "knowledge_recall"
    negative_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateLedgerEntry:
    """Provenance record for one self-edit candidate evaluation."""

    candidate_id: str
    directive_id: str
    status: str
    accepted: bool
    baseline_fitness: float
    candidate_fitness: float
    reward: float
    reasons: List[str]
    context_hash: str
    diagnostics_hash: str
    directive_hash: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    quantization_gate: Optional[Dict[str, Any]] = None
    safety: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DirectiveExecutionResult:
    """Sandbox execution record for one directive."""

    directive: SelfEditDirective
    fitness: FitnessEvaluation
    isolated: bool
    mutation_scope: str
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directive": self.directive.to_dict(),
            "fitness": {
                "candidate_id": self.fitness.candidate_id,
                "fitness": self.fitness.fitness,
                "metrics": self.fitness.metrics,
                "metadata": self.fitness.metadata,
            },
            "isolated": self.isolated,
            "mutation_scope": self.mutation_scope,
            "artifacts": self.artifacts,
        }


@dataclass
class AdaptiveValidationRound:
    """One loop of self-healing candidate execution and gate review."""

    round_index: int
    accepted_count: int
    rejected_count: int
    best_directive_id: Optional[str]
    best_fitness: Optional[float]
    status: str
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelfHealingChelationConfig:
    """Configuration for SEAL/EGGROLL-inspired self-healing edit selection."""

    baseline_fitness: float = 0.0
    reward_threshold: float = 0.0
    min_retention_score: float = 0.8
    retained_gain_threshold: float = 0.8
    minimum_fp32_gain: float = 0.0
    max_directives: int = 6
    allow_persistent_update: bool = False
    include_eggroll_directive: bool = True
    include_retention_replay: bool = True
    min_structural_health: float = 0.7
    max_latency_regression_ratio: float = 1.2
    require_self_generated_probes: bool = True

    def __post_init__(self) -> None:
        if self.min_retention_score < 0 or self.min_retention_score > 1:
            raise ValueError("min_retention_score must be in [0, 1]")
        if self.retained_gain_threshold < 0:
            raise ValueError("retained_gain_threshold must be non-negative")
        if self.minimum_fp32_gain < 0:
            raise ValueError("minimum_fp32_gain must be non-negative")
        if self.max_directives < 1:
            raise ValueError("max_directives must be >= 1")
        if self.min_structural_health < 0 or self.min_structural_health > 1:
            raise ValueError("min_structural_health must be in [0, 1]")
        if self.max_latency_regression_ratio <= 0:
            raise ValueError("max_latency_regression_ratio must be positive")


class CandidateProvenanceLedger:
    """Append-only candidate ledger for self-edit provenance and promotion review."""

    def __init__(self):
        self.entries: List[CandidateLedgerEntry] = []

    def record(
        self,
        evaluation: SelfEditEvaluation,
        baseline_fitness: float,
        context_hash: str,
        diagnostics_hash: str,
        safety: Dict[str, Any],
    ) -> CandidateLedgerEntry:
        entry = CandidateLedgerEntry(
            candidate_id=evaluation.fitness.candidate_id,
            directive_id=evaluation.directive.directive_id,
            status="accepted" if evaluation.accepted else "rejected",
            accepted=evaluation.accepted,
            baseline_fitness=float(baseline_fitness),
            candidate_fitness=float(evaluation.fitness.fitness),
            reward=float(evaluation.reward),
            reasons=list(evaluation.reasons),
            context_hash=context_hash,
            diagnostics_hash=diagnostics_hash,
            directive_hash=_stable_hash(evaluation.directive.to_dict()),
            metrics=dict(evaluation.fitness.metrics),
            quantization_gate=evaluation.quantization_gate,
            safety=dict(safety),
        )
        self.entries.append(entry)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_count": len(self.entries),
            "accepted_count": sum(1 for entry in self.entries if entry.accepted),
            "rejected_count": sum(1 for entry in self.entries if not entry.accepted),
            "entries": [entry.to_dict() for entry in self.entries],
        }


class SelfEditSandboxExecutor:
    """Execute directives in isolated candidate state and return comparable artifacts."""

    def __init__(
        self,
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
        execution_fn: Optional[Callable[[SelfEditDirective], Any]] = None,
        logger=None,
    ):
        self.fitness = fitness
        self.execution_fn = execution_fn
        self.logger = logger or get_logger()

    def execute(self, directives: Iterable[SelfEditDirective]) -> List[DirectiveExecutionResult]:
        results = []
        for directive in directives:
            fitness_result = self._execute_one(directive)
            results.append(
                DirectiveExecutionResult(
                    directive=directive,
                    fitness=fitness_result,
                    isolated=True,
                    mutation_scope=str(directive.optimization_params.get("scope", "adapter_only")),
                    artifacts={
                        "sandbox": True,
                        "base_model_mutation_allowed": False,
                        "execution_mode": directive.adaptation_mode,
                    },
                )
            )
        self.logger.log_event(
            "self_edit_sandbox_executed",
            "Executed self-edit directives in sandbox mode",
            total=len(results),
            level="DEBUG",
        )
        return results

    def _execute_one(self, directive: SelfEditDirective) -> FitnessEvaluation:
        if self.execution_fn is not None:
            result = self.execution_fn(directive)
        elif isinstance(self.fitness, FitnessFunctionInterface):
            result = self.fitness.evaluate_candidate(directive, candidate_id=directive.directive_id)
        else:
            result = self.fitness(directive)
        if isinstance(result, FitnessEvaluation):
            return result
        return FitnessEvaluation(candidate_id=directive.directive_id, fitness=float(result))


class SelfHealingChelationPlanner:
    """Generate, evaluate, and filter candidate self-healing directives.

    This is the ChelatedAI adaptation of two research ideas:
    - SEAL: generate self-edits and reinforce only those that improve downstream outcomes.
    - EGGROLL: prefer low-rank, quantization-aware, scalar-fitness updates for adapter-only optimization.
    """

    def __init__(
        self,
        config: Optional[SelfHealingChelationConfig] = None,
        logger=None,
    ):
        self.config = config or SelfHealingChelationConfig()
        self.logger = logger or get_logger()
        self.quantization_gate = QuantizationPromotionGate(
            retained_gain_threshold=self.config.retained_gain_threshold,
            minimum_fp32_gain=self.config.minimum_fp32_gain,
            logger=self.logger,
        )

    def generate_directives(
        self,
        context: Union[str, Iterable[str]],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[SelfEditDirective]:
        """Generate deterministic self-edit candidates from new context and diagnostics."""

        normalized_context = self._normalize_context(context)
        diagnostics = diagnostics or {}
        directives: List[SelfEditDirective] = []

        synthetic_examples = self._build_synthetic_examples(normalized_context)
        directives.append(
            SelfEditDirective(
                directive_id="seal_implication_sft",
                strategy="implication_synthesis",
                adaptation_mode="adapter_sft",
                synthetic_examples=synthetic_examples,
                optimization_params={
                    "loss": "causal_or_contrastive_sft",
                    "epochs": 1,
                    "learning_rate": 1e-4,
                    "scope": "adapter_only",
                },
                metadata={
                    "paper_alignment": "SEAL knowledge incorporation",
                    "persistent_update_allowed": self.config.allow_persistent_update,
                },
            )
        )

        if self._has_retrieval_anomaly(diagnostics):
            directives.append(
                SelfEditDirective(
                    directive_id="seal_retrieval_ttt",
                    strategy="retrieval_test_time_training",
                    adaptation_mode="online_contrastive_update",
                    synthetic_examples=synthetic_examples,
                    optimization_params={
                        "loss": "infonce",
                        "temperature": 0.07,
                        "epochs": 1,
                        "scope": "adapter_only",
                    },
                    metadata={"paper_alignment": "SEAL few-shot TTT"},
                )
            )

        if self.config.include_eggroll_directive:
            directives.append(
                SelfEditDirective(
                    directive_id="eggroll_low_rank_self_edit",
                    strategy="low_rank_population_search",
                    adaptation_mode="eggroll_es",
                    synthetic_examples=synthetic_examples,
                    optimization_params={
                        "optimizer": "eggroll_es",
                        "rank": 1,
                        "population_size": 16,
                        "sigma": 0.01,
                        "fitness": "retrieval_plus_structural_health",
                        "quantization_aware": True,
                        "scope": "adapter_only",
                    },
                    source="eggroll",
                    metadata={"paper_alignment": "Evolution Strategies at the Hyperscale"},
                )
            )

        if self._has_quantization_failure(diagnostics):
            directives.append(
                SelfEditDirective(
                    directive_id="quantization_survival_self_edit",
                    strategy="quantization_survival_rewrite",
                    adaptation_mode="quantization_gate",
                    synthetic_examples=synthetic_examples,
                    optimization_params={
                        "retained_gain_threshold": self.config.retained_gain_threshold,
                        "minimum_fp32_gain": self.config.minimum_fp32_gain,
                        "scope": "adapter_only",
                    },
                    source="seal+eggroll",
                    metadata={"paper_alignment": "SEAL reward filtering plus EGGROLL int8 scoring"},
                )
            )

        structural_health = diagnostics.get("structural_health")
        if isinstance(structural_health, dict) and float(structural_health.get("score", 1.0)) < 0.7:
            directives.append(
                SelfEditDirective(
                    directive_id="structural_repair_self_edit",
                    strategy="collapse_repair_replay",
                    adaptation_mode="structural_health_penalty",
                    synthetic_examples=synthetic_examples,
                    optimization_params={
                        "collapse_penalty": 0.4,
                        "isomer_penalty": 0.3,
                        "topology_penalty": 0.3,
                        "scope": "adapter_only",
                    },
                    metadata={"paper_alignment": "SEAL reward shaping for retention and collapse repair"},
                )
            )

        if self.config.include_retention_replay:
            directives.append(
                SelfEditDirective(
                    directive_id="retention_replay_guard",
                    strategy="catastrophic_forgetting_guard",
                    adaptation_mode="retention_replay",
                    synthetic_examples=synthetic_examples[:2],
                    optimization_params={
                        "min_retention_score": self.config.min_retention_score,
                        "reject_on_forgetting": True,
                    },
                    metadata={"paper_alignment": "SEAL catastrophic-forgetting limitation"},
                )
            )

        return directives[: self.config.max_directives]

    def evaluate_directives(
        self,
        directives: Iterable[SelfEditDirective],
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
    ) -> List[SelfEditEvaluation]:
        """Evaluate directives and accept only positive, gate-safe self-edits."""

        evaluations: List[SelfEditEvaluation] = []
        for index, directive in enumerate(directives):
            candidate_id = directive.directive_id or f"self_edit_{index}"
            fitness_result = self._evaluate_fitness(fitness, directive, candidate_id)
            reward = float(fitness_result.fitness) - float(self.config.baseline_fitness)
            reasons = []
            accepted = reward > self.config.reward_threshold

            if not accepted:
                reasons.append("reward_not_positive")

            retention_score = self._extract_retention_score(fitness_result)
            if retention_score is not None and retention_score < self.config.min_retention_score:
                accepted = False
                reasons.append("retention_below_threshold")

            quantization_gate = self._evaluate_quantization_gate(fitness_result)
            if quantization_gate is not None and not quantization_gate["passed"]:
                accepted = False
                reasons.append("quantization_gate_failed")

            evaluations.append(
                SelfEditEvaluation(
                    directive=directive,
                    fitness=fitness_result,
                    reward=reward,
                    accepted=accepted,
                    reasons=reasons,
                    quantization_gate=quantization_gate,
                )
            )

        self.logger.log_event(
            "self_healing_self_edits_evaluated",
            "Evaluated SEAL-style self-healing directives",
            total=len(evaluations),
            accepted=sum(1 for result in evaluations if result.accepted),
            level="DEBUG",
        )
        return evaluations

    def build_update_plan(
        self,
        context: Union[str, Iterable[str]],
        diagnostics: Dict[str, Any],
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
    ) -> Dict[str, Any]:
        """Generate and filter directives into an advisory or persistent update plan."""

        directives = self.generate_directives(context, diagnostics)
        probes = self.generate_eval_probes(context)
        evaluations = self.evaluate_directives(directives, fitness)
        accepted = [result for result in evaluations if result.accepted]
        rejected = [result for result in evaluations if not result.accepted]
        best = max(evaluations, key=lambda result: result.fitness.fitness) if evaluations else None
        safety = self._safety_metadata()
        ledger = CandidateProvenanceLedger()
        context_hash = _stable_hash(self._normalize_context(context))
        diagnostics_hash = _stable_hash(diagnostics)
        for evaluation in evaluations:
            ledger.record(
                evaluation,
                baseline_fitness=self.config.baseline_fitness,
                context_hash=context_hash,
                diagnostics_hash=diagnostics_hash,
                safety=safety,
            )
        plan = {
            "mode": "persistent_update" if self.config.allow_persistent_update else "advisory",
            "baseline_fitness": self.config.baseline_fitness,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_directives": [result.to_dict() for result in accepted],
            "rejected_directives": [result.to_dict() for result in rejected],
            "best_directive_id": best.directive.directive_id if best is not None else None,
            "best_fitness": best.fitness.fitness if best is not None else None,
            "self_generated_eval_probes": [probe.to_dict() for probe in probes],
            "candidate_ledger": ledger.to_dict(),
            "safety": safety,
        }
        self.logger.log_event(
            "self_healing_update_plan_built",
            "Built self-healing chelation update plan",
            accepted_count=plan["accepted_count"],
            rejected_count=plan["rejected_count"],
            mode=plan["mode"],
            level="DEBUG",
        )
        return plan

    def execute_shadow_round(
        self,
        context: Union[str, Iterable[str]],
        diagnostics: Dict[str, Any],
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
        execution_fn: Optional[Callable[[SelfEditDirective], Any]] = None,
    ) -> Dict[str, Any]:
        """Run one sandboxed directive round and return execution plus gated plan artifacts."""

        directives = self.generate_directives(context, diagnostics)
        executor = SelfEditSandboxExecutor(fitness=fitness, execution_fn=execution_fn, logger=self.logger)
        execution_results = executor.execute(directives)
        by_id = {result.directive.directive_id: result.fitness for result in execution_results}
        plan = self.build_update_plan(
            context=context,
            diagnostics=diagnostics,
            fitness=lambda directive: by_id[directive.directive_id],
        )
        plan["shadow_execution"] = {
            "executed_count": len(execution_results),
            "isolated": True,
            "results": [result.to_dict() for result in execution_results],
        }
        return plan

    def run_adaptive_validation_loop(
        self,
        context: Union[str, Iterable[str]],
        diagnostics: Dict[str, Any],
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
        rounds: int = 3,
        execution_fn: Optional[Callable[[SelfEditDirective], Any]] = None,
    ) -> Dict[str, Any]:
        """Run repeated sandbox/gate loops and expose debugging actions per round."""

        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        loop_diagnostics = dict(diagnostics)
        round_records: List[AdaptiveValidationRound] = []
        plans = []
        for round_index in range(1, rounds + 1):
            plan = self.execute_shadow_round(
                context=context,
                diagnostics=loop_diagnostics,
                fitness=fitness,
                execution_fn=execution_fn,
            )
            plans.append(plan)
            actions = []
            if plan["accepted_count"] == 0:
                actions.append("increase_probe_coverage")
                actions.append("keep_advisory_mode")
                loop_diagnostics["retrieval_policy"] = {"high_variance_fast_path": True}
            else:
                actions.append("retain_best_candidate_for_shadow_replay")
            round_records.append(
                AdaptiveValidationRound(
                    round_index=round_index,
                    accepted_count=plan["accepted_count"],
                    rejected_count=plan["rejected_count"],
                    best_directive_id=plan["best_directive_id"],
                    best_fitness=plan["best_fitness"],
                    status="pass" if plan["accepted_count"] > 0 else "debug",
                    actions=actions,
                )
            )
        accepted_total = sum(record.accepted_count for record in round_records)
        return {
            "rounds_requested": rounds,
            "rounds_completed": len(round_records),
            "accepted_total": accepted_total,
            "status": "pass" if accepted_total > 0 else "needs_debugging",
            "rounds": [record.to_dict() for record in round_records],
            "plans": plans,
        }

    def generate_eval_probes(self, context: Union[str, Iterable[str]]) -> List[SelfGeneratedEvalProbe]:
        """Generate deterministic SEAL-style evaluation probes with confuser negatives."""

        normalized_context = self._normalize_context(context)
        probes = []
        for index, item in enumerate(normalized_context):
            terms = _keyword_terms(item)
            probes.append(
                SelfGeneratedEvalProbe(
                    probe_id=f"knowledge_probe_{index + 1}",
                    question=f"What should the self-healing adapter preserve from context {index + 1}?",
                    expected_terms=terms,
                    probe_type="knowledge_recall",
                    negative_terms=["unrelated", "collapse", "regression"],
                )
            )
            probes.append(
                SelfGeneratedEvalProbe(
                    probe_id=f"retention_probe_{index + 1}",
                    question=f"Which prior retrieval behavior must remain stable after context {index + 1}?",
                    expected_terms=terms[:2] or ["retrieval"],
                    probe_type="retention",
                    negative_terms=["forgetting", "route drift", "quantization loss"],
                )
            )
        return probes[:8]

    def _normalize_context(self, context: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(context, str):
            items = [context]
        else:
            items = list(context)
        normalized = [" ".join(str(item).split()) for item in items if str(item).strip()]
        if not normalized:
            raise ValueError("context must include at least one non-empty item")
        return normalized

    def _build_synthetic_examples(self, context: List[str]) -> List[str]:
        examples = []
        for index, item in enumerate(context):
            examples.append(f"Implication {index + 1}: {item}")
            examples.append(f"Retrieval QA {index + 1}: What should remain recoverable? {item}")
        return examples[:8]

    def _has_retrieval_anomaly(self, diagnostics: Dict[str, Any]) -> bool:
        runtime = diagnostics.get("runtime")
        retrieval_policy = diagnostics.get("retrieval_policy")
        if isinstance(runtime, dict) and runtime.get("status") in {"empty_results", "qdrant_error"}:
            return True
        if isinstance(retrieval_policy, dict) and retrieval_policy.get("high_variance_fast_path"):
            return True
        return False

    def _has_quantization_failure(self, diagnostics: Dict[str, Any]) -> bool:
        quantization_gate = diagnostics.get("quantization_gate")
        return isinstance(quantization_gate, dict) and quantization_gate.get("passed") is False

    def _extract_retention_score(self, fitness_result: FitnessEvaluation) -> Optional[float]:
        for container in (fitness_result.metrics, fitness_result.metadata):
            value = container.get("retention_score")
            if value is not None:
                return float(value)
        return None

    def _evaluate_quantization_gate(self, fitness_result: FitnessEvaluation) -> Optional[Dict[str, Any]]:
        fp32_fitness = fitness_result.metrics.get("fp32_fitness")
        quantized_fitness = fitness_result.metrics.get("quantized_fitness")
        if fp32_fitness is None or quantized_fitness is None:
            return None
        return self.quantization_gate.evaluate(
            fp32_fitness=float(fp32_fitness),
            quantized_fitness=float(quantized_fitness),
            baseline_fitness=self.config.baseline_fitness,
        ).to_dict()

    def _evaluate_fitness(
        self,
        fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
        directive: SelfEditDirective,
        candidate_id: str,
    ) -> FitnessEvaluation:
        if isinstance(fitness, FitnessFunctionInterface):
            return fitness.evaluate_candidate(directive, candidate_id=candidate_id)

        result = fitness(directive)
        if isinstance(result, FitnessEvaluation):
            return result
        return FitnessEvaluation(candidate_id=candidate_id, fitness=float(result))

    def _safety_metadata(self) -> Dict[str, Any]:
        return {
            "base_model_mutation_allowed": False,
            "adapter_only": True,
            "retention_gate": self.config.min_retention_score,
            "quantization_retained_gain_gate": self.config.retained_gain_threshold,
            "min_structural_health": self.config.min_structural_health,
            "max_latency_regression_ratio": self.config.max_latency_regression_ratio,
            "persistent_update_enabled": self.config.allow_persistent_update,
        }


def build_self_healing_update_plan(
    context: Union[str, Iterable[str]],
    diagnostics: Dict[str, Any],
    fitness: Union[FitnessFunctionInterface, Callable[[SelfEditDirective], Any]],
    config: Optional[SelfHealingChelationConfig] = None,
    logger=None,
) -> Dict[str, Any]:
    """Convenience wrapper for one-shot self-healing update-plan generation."""

    return SelfHealingChelationPlanner(config=config, logger=logger).build_update_plan(
        context=context,
        diagnostics=diagnostics,
        fitness=fitness,
    )


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _keyword_terms(text: str, limit: int = 5) -> List[str]:
    tokens = []
    for raw in str(text).lower().replace("-", " ").replace("_", " ").split():
        token = "".join(character for character in raw if character.isalnum())
        if len(token) >= 4 and token not in tokens:
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tokens or ["retrieval"]

