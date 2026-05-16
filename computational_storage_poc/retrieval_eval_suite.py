from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    expected_path_suffix: str


RETRIEVAL_EVAL_CASES: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase(
        query="packed graph artifact manifest mmap int8 scales",
        expected_path_suffix="packed_graph.py",
    ),
    RetrievalEvalCase(
        query="sparse selective loading chunk cache stream from block",
        expected_path_suffix="sparse_cpu_inference.py",
    ),
    RetrievalEvalCase(
        query="prequantized int8 cpu backend matmul quantized weights",
        expected_path_suffix="cpu_backends.py",
    ),
    RetrievalEvalCase(
        query="train digits classifier compile packed int8 model",
        expected_path_suffix="train_and_compile.py",
    ),
    RetrievalEvalCase(
        query="repo graph memory query api local imports symbols",
        expected_path_suffix="repo_graph_memory.py",
    ),
    RetrievalEvalCase(
        query="integrated repo runtime reranker retrieval bytes read",
        expected_path_suffix="integrated_repo_runtime.py",
    ),
    RetrievalEvalCase(
        query="moe reap expert pruning routed expert execution",
        expected_path_suffix="moe_reap.py",
    ),
    RetrievalEvalCase(
        query="deterministic trigger sector payload usb transport contract",
        expected_path_suffix="payload_contract.py",
    ),
    RetrievalEvalCase(
        query="virtual controller emulation path sector reads",
        expected_path_suffix="emulation/virtual_controller.py",
    ),
    RetrievalEvalCase(
        query="disk llm estimator sparse flash tokens per second",
        expected_path_suffix="disk_llm_estimator.py",
    ),
)


def get_retrieval_eval_cases() -> tuple[RetrievalEvalCase, ...]:
    return RETRIEVAL_EVAL_CASES
