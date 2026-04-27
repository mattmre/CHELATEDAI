"""Low-rank zeroth-order optimization for ChelatedAI adapters.

This module implements an EGGROLL-inspired optimizer for adapter-only
correction layers. It uses low-rank parameter perturbations, scalar fitness
evaluation, deterministic seed replay, optional quantization-aware scoring, and
Kalman-style perturbation-scale adaptation. It is intentionally opt-in and does
not mutate base embedding models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from chelation_logger import get_logger
from elite_archive import EliteArchive
from fitness_interfaces import FitnessFunctionInterface
from quantization_promotion_gate import QuantizationPromotionGate
from reproducibility_context import ReproducibilityContext


FitnessFn = Callable[[], float]


@dataclass
class EvolutionStrategiesConfig:
    """Configuration for low-rank Evolution Strategies optimization."""

    population_size: int = 16
    rank: int = 1
    sigma: float = 0.01
    learning_rate: float = 0.01
    generations: int = 5
    seed: int = 0
    normalize_fitness: bool = True
    quantization_aware: bool = False
    quantization_levels: int = 127
    kalman_sigma: bool = False
    kalman_process_noise: float = 0.1
    min_sigma_ratio: float = 0.25
    max_sigma_ratio: float = 2.0
    elite_pool_size: int = 3
    rollback_to_elite: bool = False
    antithetic_sampling: bool = False
    fitness_shaping: str = "zscore"  # "zscore", "centered", or "linear_rank"
    storage_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.quantization_levels < 2:
            raise ValueError("quantization_levels must be >= 2")
        if self.kalman_process_noise <= 0:
            raise ValueError("kalman_process_noise must be positive")
        if self.min_sigma_ratio <= 0:
            raise ValueError("min_sigma_ratio must be positive")
        if self.max_sigma_ratio < self.min_sigma_ratio:
            raise ValueError("max_sigma_ratio must be >= min_sigma_ratio")
        if self.elite_pool_size < 1:
            raise ValueError("elite_pool_size must be >= 1")
        if self.antithetic_sampling and self.population_size < 2:
            raise ValueError("population_size must be >= 2 for antithetic_sampling")
        if self.fitness_shaping not in {"zscore", "centered", "linear_rank"}:
            raise ValueError("fitness_shaping must be 'zscore', 'centered', or 'linear_rank'")


def simulate_int8_quantization(
    tensor: torch.Tensor,
    levels: int = 127,
    quantile: float = 0.99,
) -> torch.Tensor:
    """Simulate symmetric scalar INT8 dequantization for fitness scoring."""

    if levels < 2:
        raise ValueError("levels must be >= 2")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")

    if tensor.numel() == 0:
        return tensor.clone()

    abs_values = tensor.detach().abs().reshape(-1)
    scale_ref = torch.quantile(abs_values, quantile)
    if scale_ref <= 1e-12:
        return torch.zeros_like(tensor)

    scale = scale_ref / float(levels)
    quantized = torch.clamp(torch.round(tensor / scale), -levels, levels)
    return quantized * scale


class KalmanSigmaScheduler:
    """Kalman-gain-inspired perturbation-scale scheduler for ES fitness variance."""

    def __init__(
        self,
        base_sigma: float,
        process_noise: float = 0.1,
        min_sigma_ratio: float = 0.25,
        max_sigma_ratio: float = 2.0,
    ):
        if base_sigma <= 0:
            raise ValueError("base_sigma must be positive")
        if process_noise <= 0:
            raise ValueError("process_noise must be positive")
        if min_sigma_ratio <= 0:
            raise ValueError("min_sigma_ratio must be positive")
        if max_sigma_ratio < min_sigma_ratio:
            raise ValueError("max_sigma_ratio must be >= min_sigma_ratio")

        self.base_sigma = base_sigma
        self.process_noise = process_noise
        self.min_sigma = base_sigma * min_sigma_ratio
        self.max_sigma = base_sigma * max_sigma_ratio
        self.current_sigma = base_sigma
        self.kalman_gain = 1.0

    def step(self, fitness_values: List[float]) -> float:
        """Update sigma from the variance of one population's fitness values."""

        if len(fitness_values) < 2:
            return self.current_sigma

        measurement_noise = float(np.var(fitness_values))
        self.kalman_gain = self.process_noise / (self.process_noise + measurement_noise + 1e-10)
        proposed = self.base_sigma * self.kalman_gain
        self.current_sigma = max(self.min_sigma, min(self.max_sigma, proposed))
        return self.current_sigma

    def get_state(self) -> Dict[str, float]:
        return {
            "base_sigma": self.base_sigma,
            "current_sigma": self.current_sigma,
            "kalman_gain": self.kalman_gain,
            "process_noise": self.process_noise,
        }


class LowRankEvolutionStrategyOptimizer:
    """EGGROLL-style low-rank ES optimizer for adapter parameters."""

    def __init__(
        self,
        module: nn.Module,
        config: Optional[EvolutionStrategiesConfig] = None,
        logger: Optional[Any] = None,
    ):
        self.module = module
        self.config = config or EvolutionStrategiesConfig()
        self.logger = logger or get_logger()
        self.params = [p for p in module.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("module has no trainable parameters")

        self._base_seed = int(self.config.seed)
        self._step_count = 0
        self._elite_archive = EliteArchive(max_size=self.config.elite_pool_size)
        self._reproducibility = ReproducibilityContext.create(
            optimizer_type="eggroll_es",
            config=asdict(self.config),
            seed=self.config.seed,
        )
        self._sigma_scheduler = (
            KalmanSigmaScheduler(
                base_sigma=self.config.sigma,
                process_noise=self.config.kalman_process_noise,
                min_sigma_ratio=self.config.min_sigma_ratio,
                max_sigma_ratio=self.config.max_sigma_ratio,
            )
            if self.config.kalman_sigma
            else None
        )

    @property
    def current_sigma(self) -> float:
        if self._sigma_scheduler is None:
            return self.config.sigma
        return self._sigma_scheduler.current_sigma

    def _make_generator(self, seed: int, device: torch.device) -> torch.Generator:
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _sample_parameter_perturbation(
        self,
        param: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if param.dim() == 2:
            rows, cols = param.shape
            rank = min(self.config.rank, rows, cols)
            left = torch.randn(rows, rank, generator=generator, device=param.device, dtype=param.dtype)
            right = torch.randn(cols, rank, generator=generator, device=param.device, dtype=param.dtype)
            return (left @ right.t()) / sqrt(float(rank))

        return torch.randn(param.shape, generator=generator, device=param.device, dtype=param.dtype)

    def _sample_population_member(self, member_index: int) -> List[torch.Tensor]:
        perturbations = []
        seed_base = self._base_seed + self._step_count * 1000003 + member_index * 9176
        for param_index, param in enumerate(self.params):
            generator = self._make_generator(seed_base + param_index, param.device)
            perturbations.append(self._sample_parameter_perturbation(param, generator))
        return perturbations

    def _sample_population(self) -> List[List[torch.Tensor]]:
        if not self.config.antithetic_sampling:
            return [self._sample_population_member(index) for index in range(self.config.population_size)]

        base_count = (self.config.population_size + 1) // 2
        sampled = [self._sample_population_member(index) for index in range(base_count)]
        population = list(sampled)
        for perturbations in sampled:
            if len(population) >= self.config.population_size:
                break
            population.append([-perturbation for perturbation in perturbations])
        return population

    def _snapshot(self) -> List[torch.Tensor]:
        return [param.detach().clone() for param in self.params]

    def _restore(self, snapshot: List[torch.Tensor]) -> None:
        with torch.no_grad():
            for param, value in zip(self.params, snapshot):
                param.copy_(value)

    def _apply_perturbation(self, base: List[torch.Tensor], perturbations: List[torch.Tensor], sigma: float) -> None:
        with torch.no_grad():
            for param, base_value, perturbation in zip(self.params, base, perturbations):
                param.copy_(base_value + sigma * perturbation)

    def _candidate_parameters(
        self,
        base: List[torch.Tensor],
        perturbations: List[torch.Tensor],
        sigma: float,
    ) -> List[torch.Tensor]:
        return [
            base_value.detach().clone() + sigma * perturbation.detach().clone()
            for base_value, perturbation in zip(base, perturbations)
        ]

    def _shape_fitness(self, fitness_tensor: torch.Tensor) -> torch.Tensor:
        if self.config.fitness_shaping == "linear_rank":
            order = torch.argsort(fitness_tensor, descending=False)
            ranks = torch.empty_like(fitness_tensor)
            ranks[order] = torch.arange(len(fitness_tensor), dtype=torch.float32, device=fitness_tensor.device)
            centered = ranks - ranks.mean()
        else:
            centered = fitness_tensor - fitness_tensor.mean()

        if self.config.fitness_shaping == "centered":
            return centered

        std = centered.std(unbiased=False)
        if std > 1e-12:
            centered = centered / std
        return centered

    def _evaluate_fitness(self, fitness_fn: Any, candidate_id: str) -> float:
        if isinstance(fitness_fn, FitnessFunctionInterface):
            return float(fitness_fn.evaluate_candidate(self.module, candidate_id=candidate_id).fitness)
        return float(fitness_fn())

    def step(self, fitness_fn: FitnessFn) -> Dict[str, Any]:
        """Run one ES generation and apply the weighted perturbation update."""

        self.module.eval()
        base = self._snapshot()
        sigma = self.current_sigma
        population_perturbations: List[List[torch.Tensor]] = []
        fitness_values: List[float] = []

        try:
            for member_index, perturbations in enumerate(self._sample_population()):
                population_perturbations.append(perturbations)
                self._apply_perturbation(base, perturbations, sigma)
                fitness_values.append(self._evaluate_fitness(fitness_fn, f"generation_{self._step_count + 1}_member_{member_index}"))
        finally:
            self._restore(base)

        fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32)
        centered = self._shape_fitness(fitness_tensor)
        if not self.config.normalize_fitness and self.config.fitness_shaping == "zscore":
            centered = fitness_tensor - fitness_tensor.mean()

        top_count = min(self.config.elite_pool_size, len(fitness_values))
        top_indices = torch.topk(fitness_tensor, k=top_count).indices.tolist()
        for rank_index, member_index in enumerate(top_indices):
            self._elite_archive.add_candidate(
                candidate_id=f"generation_{self._step_count + 1}_member_{member_index}",
                fitness=fitness_values[member_index],
                generation=self._step_count + 1,
                parameters=self._candidate_parameters(base, population_perturbations[member_index], sigma),
                metadata={
                    "member_index": member_index,
                    "rank": rank_index + 1,
                    "sigma": sigma,
                    "antithetic_sampling": self.config.antithetic_sampling,
                    "fitness_shaping": self.config.fitness_shaping,
                },
            )

        with torch.no_grad():
            for param_index, param in enumerate(self.params):
                update = torch.zeros_like(param)
                for member_index, perturbations in enumerate(population_perturbations):
                    update = update + centered[member_index].to(param.device, param.dtype) * perturbations[param_index]
                update = update / float(self.config.population_size)
                param.copy_(base[param_index] + self.config.learning_rate * update)

        final_update_fitness = None
        restored_elite = False
        if self.config.rollback_to_elite:
            final_update_fitness = self._evaluate_fitness(fitness_fn, f"generation_{self._step_count + 1}_final")
            best = self._elite_archive.best()
            if best is not None and best.fitness > final_update_fitness:
                restored_elite = self._elite_archive.restore_best(self.params)

        if self._sigma_scheduler is not None:
            self._sigma_scheduler.step(fitness_values)

        self._step_count += 1
        storage_evaluation = None
        if self.config.storage_profile:
            from computational_storage_poc.mock_array import ArraySimulation
            from device_profiles import get_profile

            storage_evaluation = ArraySimulation(
                device_profile=get_profile(self.config.storage_profile)
            ).sharded_population_evaluation([
                {"candidate_id": f"generation_{self._step_count}_member_{index}", "fitness": fitness}
                for index, fitness in enumerate(fitness_values)
            ])
        result = {
            "generation": self._step_count,
            "mean_fitness": float(fitness_tensor.mean().item()),
            "best_fitness": float(fitness_tensor.max().item()),
            "worst_fitness": float(fitness_tensor.min().item()),
            "fitness_std": float(fitness_tensor.std(unbiased=False).item()),
            "sigma": sigma,
            "elite_candidates": self._elite_archive.summaries(),
            "restored_elite": restored_elite,
        }
        if storage_evaluation is not None:
            result["storage_evaluation"] = storage_evaluation
        if final_update_fitness is not None:
            result["final_update_fitness"] = final_update_fitness
        self.logger.log_event(
            "eggroll_es_generation",
            "Low-rank ES generation completed",
            **result,
        )
        return result

    def optimize(self, fitness_fn: FitnessFn, generations: Optional[int] = None) -> Dict[str, Any]:
        """Run multiple ES generations."""

        total_generations = generations if generations is not None else self.config.generations
        history = [self.step(fitness_fn) for _ in range(total_generations)]
        return {
            "generations": total_generations,
            "history": history,
            "final": history[-1] if history else {},
            "sigma_state": self._sigma_scheduler.get_state() if self._sigma_scheduler else None,
            "elite_candidates": self._elite_archive.summaries(),
            "reproducibility": self._reproducibility.to_dict(),
        }


def train_adapter_with_es(
    adapter: nn.Module,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    criterion: nn.Module,
    config: Optional[EvolutionStrategiesConfig] = None,
    logger: Optional[Any] = None,
    quantization_gate: Optional[QuantizationPromotionGate] = None,
) -> Dict[str, Any]:
    """Train an adapter against fixed targets using low-rank ES fitness."""

    es_config = config or EvolutionStrategiesConfig()
    optimizer = LowRankEvolutionStrategyOptimizer(adapter, es_config, logger=logger)

    with torch.no_grad():
        baseline_outputs = adapter(input_tensor)
        baseline_loss = criterion(baseline_outputs, target_tensor)
        baseline_fitness = -float(baseline_loss.item())

    def fitness_fn() -> float:
        with torch.no_grad():
            outputs = adapter(input_tensor)
            if es_config.quantization_aware:
                outputs = simulate_int8_quantization(outputs, levels=es_config.quantization_levels)
            loss = criterion(outputs, target_tensor)
            reg_loss = getattr(adapter, "regularization_loss", lambda: 0.0)()
            if isinstance(reg_loss, torch.Tensor):
                loss = loss + 0.01 * reg_loss
            elif reg_loss:
                loss = loss + 0.01 * torch.tensor(float(reg_loss), dtype=loss.dtype, device=loss.device)
            return -float(loss.item())

    result = optimizer.optimize(fitness_fn, generations=es_config.generations)
    with torch.no_grad():
        outputs = adapter(input_tensor)
        fp32_loss = criterion(outputs, target_tensor)
        if es_config.quantization_aware:
            quantized_outputs = simulate_int8_quantization(outputs, levels=es_config.quantization_levels)
        else:
            quantized_outputs = outputs
        final_loss = float(criterion(quantized_outputs, target_tensor).item())
        fp32_fitness = -float(fp32_loss.item())
        quantized_fitness = -final_loss
    result["final_loss"] = final_loss
    result["fp32_final_loss"] = float(fp32_loss.item())
    result["baseline_fitness"] = baseline_fitness
    result["fp32_fitness"] = fp32_fitness
    result["quantized_fitness"] = quantized_fitness
    if quantization_gate is not None:
        result["quantization_gate"] = quantization_gate.evaluate(
            fp32_fitness=fp32_fitness,
            quantized_fitness=quantized_fitness,
            baseline_fitness=baseline_fitness,
        ).to_dict()
    return result


class EvolutionaryOnlineUpdater:
    """Inference-time micro-population ES updater for query retrieval feedback."""

    def __init__(
        self,
        adapter: nn.Module,
        config: Optional[EvolutionStrategiesConfig] = None,
        update_interval: int = 1,
        logger: Optional[Any] = None,
    ):
        if update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        self.adapter = adapter
        self.config = config or EvolutionStrategiesConfig(population_size=8, generations=1, sigma=0.005)
        self.update_interval = update_interval
        self.logger = logger or get_logger()
        self._optimizer = LowRankEvolutionStrategyOptimizer(adapter, self.config, logger=self.logger)
        self._call_count = 0

    def update(self, query_vec: np.ndarray, positive_vecs: np.ndarray, negative_vecs: np.ndarray) -> Optional[Dict[str, Any]]:
        """Apply a micro-population update from query, positive, and negative vectors."""

        self._call_count += 1
        if self._call_count % self.update_interval != 0:
            return None
        if len(positive_vecs) == 0 or len(negative_vecs) == 0:
            raise ValueError("positive_vecs and negative_vecs must be non-empty")

        query_tensor = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0)
        positive_tensor = torch.tensor(positive_vecs, dtype=torch.float32)
        negative_tensor = torch.tensor(negative_vecs, dtype=torch.float32)

        def fitness_fn() -> float:
            with torch.no_grad():
                adapted_query = self.adapter(query_tensor)
                adapted_pos = self.adapter(positive_tensor)
                adapted_neg = self.adapter(negative_tensor)
                if self.config.quantization_aware:
                    adapted_query = simulate_int8_quantization(adapted_query, levels=self.config.quantization_levels)
                    adapted_pos = simulate_int8_quantization(adapted_pos, levels=self.config.quantization_levels)
                    adapted_neg = simulate_int8_quantization(adapted_neg, levels=self.config.quantization_levels)
                pos_sim = torch.mm(adapted_query, adapted_pos.t()).mean()
                neg_sim = torch.mm(adapted_query, adapted_neg.t()).mean()
                return float((pos_sim - neg_sim).item())

        result = self._optimizer.optimize(fitness_fn, generations=self.config.generations)
        self.logger.log_event(
            "online_eggroll_es_update",
            "Online ES micro-population update completed",
            generation_count=self.config.generations,
            population_size=self.config.population_size,
            final_fitness=result["final"].get("mean_fitness"),
        )
        return result

