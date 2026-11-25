"""
Shared utilities for benchmark scripts.

This module contains common helpers used across multiple benchmark scripts
(benchmark_rlm.py, benchmark_evolution.py, etc.) to avoid code duplication.

Functions:
    - canonicalize_id: Convert mixed ID types (int/str/UUID) to stable string keys
    - map_predicted_ids: Map Qdrant point IDs back to original document IDs
    - dcg_at_k: Discounted Cumulative Gain at rank k
    - ndcg_at_k: Normalized Discounted Cumulative Gain at rank k
    - mean_average_precision_at_k: Mean Average Precision at rank k
    - mean_reciprocal_rank: Mean Reciprocal Rank
    - recall_at_k: Recall at rank k
    - find_keys: Recursively search nested dict for keys
    - find_payload: Recursively search nested dict for a specific key's value
    - load_mteb_data: Load corpus, queries, and qrels from MTEB tasks
"""

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC, Set as SetABC
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import shutil
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from uuid import UUID, uuid4

METRIC_CONTRACT_VERSION = "mlr-v1"

try:
    import mteb
except ImportError:
    mteb = None


# =============================================================================
# ID Canonicalization
# =============================================================================

def canonicalize_id(id_val: Union[int, str, UUID]) -> str:
    """
    Convert mixed ID types (int/str/UUID) to stable string keys.
    
    This helper ensures consistent ID handling across different ID formats
    (integer, string, UUID) to prevent type mismatch issues when mapping
    between Qdrant point IDs and original document IDs.
    
    Args:
        id_val: ID value that can be int, str, or UUID
        
    Returns:
        str: Canonicalized string representation of the ID
        
    Examples:
        >>> canonicalize_id(123)
        '123'
        >>> canonicalize_id("doc_456")
        'doc_456'
        >>> from uuid import UUID
        >>> canonicalize_id(UUID('12345678-1234-5678-1234-567812345678'))
        '12345678-1234-5678-1234-567812345678'
    """
    if isinstance(id_val, UUID):
        return str(id_val)
    elif isinstance(id_val, int):
        return str(id_val)
    elif isinstance(id_val, str):
        return id_val
    else:
        # Fallback: try str() conversion for any other type
        return str(id_val)


def map_predicted_ids(engine, pred_ids):
    """
    Map engine/Qdrant internal IDs back to original document IDs.

    Args:
        engine: AntigravityEngine-like object with qdrant client
        pred_ids: List of internal point IDs returned by run_inference()

    Returns:
        List[str]: Original document IDs when present in payload, otherwise
        canonicalized internal IDs.
    """
    try:
        points = engine.qdrant.retrieve(engine.collection_name, ids=pred_ids)
    except Exception:
        return [canonicalize_id(pid) for pid in pred_ids]

    # Build a map from internal ID to original doc ID so we can return
    # mapped IDs in the same order as the input pred_ids list.
    id_to_original: dict = {}
    for point in points:
        internal_id = getattr(point, "id", None)
        payload = getattr(point, "payload", {}) or {}
        original_id = (
            payload.get("doc_id")
            or payload.get("original_id")
            or payload.get("id")
            or internal_id
        )
        id_to_original[internal_id] = canonicalize_id(original_id)

    return [id_to_original.get(pid, canonicalize_id(pid)) for pid in pred_ids]


@contextmanager
def isolated_adapter_state(adapter_path: Optional[Union[str, Path]] = None):
    """
    Hide the shared adapter checkpoint for an isolated benchmark run.

    Benchmarks should start from a clean adapter state and must not leave
    behind mutated checkpoint files that contaminate later configurations.
    This context manager temporarily removes the shared adapter file, then
    restores the original contents after the isolated block completes.
    """
    if adapter_path is None:
        from config import ChelationConfig

        target_path = Path(ChelationConfig.ADAPTER_WEIGHTS_PATH)
    else:
        target_path = Path(adapter_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None

    if target_path.exists():
        backup_path = target_path.with_name(
            f"{target_path.stem}.benchmark-backup-{uuid4().hex}{target_path.suffix}"
        )
        shutil.copy2(target_path, backup_path)
        target_path.unlink()

    try:
        yield target_path
    finally:
        target_path.unlink(missing_ok=True)
        if backup_path is not None and backup_path.exists():
            shutil.move(str(backup_path), str(target_path))


# =============================================================================
# Metric Calculation (Canonical Complete-Qrels Contract: mlr-v1)
# =============================================================================

def _validate_k(k: Any) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def validate_ranked_ids(ranked_ids: Iterable[Any], k: Optional[int] = None) -> List[str]:
    """Canonicalize ranked document IDs and fail closed on duplicates."""
    if isinstance(ranked_ids, (str, bytes)):
        raise TypeError("ranked_ids must be a sequence of document IDs, not a string/bytes")
    if k is not None:
        k = _validate_k(k)
        raw_items = list(ranked_ids)[:k]
    else:
        raw_items = list(ranked_ids)
    ranked = [canonicalize_id(doc_id) for doc_id in raw_items]
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_ids must not contain duplicate document IDs")
    return ranked


def _normalize_qrels_gains(qrels: Any, *, binary: bool = False) -> Dict[str, float]:
    """Normalize complete qrels (mapping or positive-ID set/sequence) to positive gains."""
    if isinstance(qrels, (int, float, bool)) or qrels is None:
        raise TypeError(
            "ndcg_at_k requires (ranked_ids, complete_qrels, k=...); "
            "legacy retrieved-only gain-vector signature ndcg_at_k(r, k) is forbidden"
        )
    if isinstance(qrels, (str, bytes)):
        raise TypeError("qrels must be a mapping or collection of document IDs, not a string")

    pos_gains: Dict[str, float] = {}
    if isinstance(qrels, MappingABC):
        for doc_id, score in qrels.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"qrel score for {doc_id!r} must be numeric")
            val = float(score)
            if not math.isfinite(val):
                raise ValueError(f"qrel score for {doc_id!r} must be finite")
            if val > 0.0:
                pos_gains[canonicalize_id(doc_id)] = 1.0 if binary else val
        return pos_gains

    if isinstance(qrels, (SetABC, SequenceABC, Iterable)):
        for doc_id in qrels:
            pos_gains[canonicalize_id(doc_id)] = 1.0
        return pos_gains

    raise TypeError("qrels must be a mapping of doc_id -> relevance or an iterable of relevant doc_ids")


def build_metric_contract_manifest(
    *,
    metric_name: str = "ndcg_at_k",
    k: int = 10,
    gain_rule: str = "binary",
    qrels_sha256: str,
    query_inclusion_rule: str = "positive_qrels_only",
    tie_break: str = "score_desc_doc_id_asc",
) -> Dict[str, Any]:
    """Return content-addressed metadata binding metric implementation and qrels."""
    k_val = _validate_k(k)
    impl_bytes = Path(__file__).read_bytes()
    impl_sha256 = hashlib.sha256(impl_bytes).hexdigest()
    return {
        "contract_version": METRIC_CONTRACT_VERSION,
        "metric_name": str(metric_name),
        "k": k_val,
        "gain_rule": str(gain_rule),
        "ideal_ranking_population": "all_positive_qrels",
        "query_inclusion_rule": str(query_inclusion_rule),
        "qrels_sha256": str(qrels_sha256),
        "implementation_file": Path(__file__).name,
        "implementation_sha256": impl_sha256,
        "tie_break": str(tie_break),
        "duplicate_doc_id_policy": "fail_closed_value_error",
    }


def dcg_at_k(r, k):
    """
    Discounted Cumulative Gain at rank k.

    Args:
        r: Array-like of relevance scores (binary or graded)
        k: Rank cutoff

    Returns:
        float: DCG score
    """
    k = _validate_k(k)
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0


def ndcg_at_k(ranked_ids, qrels, k=10, *, binary: bool = False) -> float:
    """
    Normalized Discounted Cumulative Gain at rank k using ALL positive qrels for IDCG.

    Fails closed on legacy retrieved-only gain-vector calls (ndcg_at_k(gains, k)),
    duplicate ranked document IDs, and invalid cutoffs.

    Args:
        ranked_ids: Ordered sequence of retrieved document IDs.
        qrels: Complete positive-qrel set/iterable (binary) or mapping {doc_id: score}
            (graded unless binary=True).
        k: Rank cutoff (>= 1).
        binary: If True, binarize all positive mapping scores (> 0) to 1.0.

    Returns:
        float: NDCG@k in [0.0, 1.0].
    """
    k = _validate_k(k)
    pos_gains = _normalize_qrels_gains(qrels, binary=binary)
    ranked = validate_ranked_ids(ranked_ids, k=k)
    if not pos_gains:
        return 0.0

    dcg = sum(
        pos_gains[doc_id] / math.log2(rank + 2.0)
        for rank, doc_id in enumerate(ranked)
        if doc_id in pos_gains
    )
    ideal_gains = sorted(pos_gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2.0) for rank, gain in enumerate(ideal_gains))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


# =============================================================================
# Extended Metrics (Phase 6)
# =============================================================================

def mean_average_precision_at_k(retrieved_ids, relevant_ids, k=10) -> float:
    """
    Average Precision at rank k for a single query.
    Fails closed on duplicate document IDs so AP@k is strictly bounded in [0.0, 1.0].
    """
    k = _validate_k(k)
    retrieved = validate_ranked_ids(retrieved_ids, k=k)
    relevant_set = set(_normalize_qrels_gains(relevant_ids, binary=True).keys())

    if not relevant_set:
        return 0.0

    num_relevant = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    return float(sum_precision / min(len(relevant_set), k))


def mean_reciprocal_rank(retrieved_ids, relevant_ids) -> float:
    """
    Reciprocal Rank for a single query.
    Fails closed on duplicate document IDs.
    """
    retrieved = validate_ranked_ids(retrieved_ids, k=None)
    relevant_set = set(_normalize_qrels_gains(relevant_ids, binary=True).keys())

    if not relevant_set:
        return 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return float(1.0 / (i + 1))

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k=10) -> float:
    """
    Recall at rank k for a single query.
    Fails closed on duplicate document IDs.
    """
    k = _validate_k(k)
    retrieved_set = set(validate_ranked_ids(retrieved_ids, k=k))
    relevant_set = set(_normalize_qrels_gains(relevant_ids, binary=True).keys())

    if not relevant_set:
        return 0.0

    return float(len(retrieved_set & relevant_set) / len(relevant_set))


# =============================================================================
# MTEB Data Loading Helpers
# =============================================================================

def find_keys(obj, target_keys):
    """
    Recursively search a nested dict for a level containing all target_keys.
    
    Args:
        obj: Object to search (typically a nested dict)
        target_keys: List of keys that must all be present
        
    Returns:
        dict or None: First dict containing all target_keys, or None if not found
    """
    if not isinstance(obj, dict):
        return None
    if all(k in obj for k in target_keys):
        return obj
    for k, v in obj.items():
        found = find_keys(v, target_keys)
        if found:
            return found
    return None


def find_payload(obj, key):
    """
    Recursively search a nested dict for a specific key and return its value.
    
    Args:
        obj: Object to search (typically a nested dict)
        key: Key to find
        
    Returns:
        Value associated with key, or None if not found
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            res = find_payload(v, key)
            if res is not None:
                return res
    return None


def load_mteb_data(task_name: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load corpus, queries, and qrels from an MTEB retrieval task.
    
    This function handles various MTEB task formats and provides robust
    extraction of corpus documents, queries, and relevance judgments.
    
    Args:
        task_name: Name of the MTEB task (e.g., "SciFact", "NFCorpus")
        
    Returns:
        Tuple of (corpus, queries, qrels) dicts, or (None, None, None) on error
        - corpus: {doc_id: text} mapping
        - queries: {query_id: text} mapping  
        - qrels: {query_id: {doc_id: relevance_score}} mapping
    """
    if mteb is None:
        print("ERROR: mteb package not installed. Run: pip install mteb")
        return None, None, None
    
    try:
        task = mteb.get_task(task_name)
    except KeyError:
        print(f"ERROR: Task '{task_name}' not found in MTEB registry!")
        return None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load MTEB task '{task_name}': {e}")
        return None, None, None

    task.load_data()

    # Try to find the data root containing corpus and queries
    targets = ['corpus', 'queries']
    data_root = find_keys(task.dataset, targets)

    if not data_root:
        c_payload = find_payload(task.dataset, 'corpus')
        q_payload = find_payload(task.dataset, 'queries')
        r_payload = find_payload(task.dataset, 'relevant_docs')
        if not r_payload:
            r_payload = find_payload(task.dataset, 'test')
    else:
        c_payload = data_root.get('corpus')
        q_payload = data_root.get('queries')
        r_payload = data_root.get('relevant_docs')

    corpus = {}
    queries = {}
    qrels = {}

    # Parse corpus
    if c_payload:
        try:
            for k, v in c_payload.items():
                corpus[k] = v['text'] + " " + v['title']
        except (AttributeError, TypeError):
            for row in c_payload:
                if '_id' in row:
                    doc_id = row['_id']
                elif 'id' in row:
                    doc_id = row['id']
                else:
                    continue
                text = row.get('text', '')
                title = row.get('title', '')
                corpus[doc_id] = text + " " + title
        except Exception as e:
            print(f"ERROR: Failed to parse corpus payload: {e}")

    # Parse queries
    if q_payload:
        try:
            for k, v in q_payload.items():
                queries[k] = v['text']
        except (AttributeError, TypeError):
            for row in q_payload:
                if '_id' in row:
                    qid = row['_id']
                elif 'id' in row:
                    qid = row['id']
                else:
                    continue
                queries[qid] = row.get('text', '')
        except Exception as e:
            print(f"ERROR: Failed to parse queries payload: {e}")

    # Parse qrels (relevance judgments)
    if r_payload:
        try:
            if isinstance(r_payload, dict):
                for qid, docs in r_payload.items():
                    qid = str(qid)
                    qrels[qid] = {}
                    if isinstance(docs, dict):
                        for did, score in docs.items():
                            qrels[qid][str(did)] = score
                    else:
                        for did in docs:
                            qrels[qid][str(did)] = 1
            else:
                for row in r_payload:
                    qid = str(row.get('query-id', row.get('query_id', row.get('_id'))))
                    if not qid or qid == 'None':
                        continue
                    if qid not in qrels:
                        qrels[qid] = {}
                    if 'doc-ids' in row:
                        for did in row['doc-ids']:
                            qrels[qid][str(did)] = 1
                    elif 'doc_ids' in row:
                        for did in row['doc_ids']:
                            qrels[qid][str(did)] = 1
                    elif 'doc-id' in row:
                        qrels[qid][str(row['doc-id'])] = row.get('score', 1)
                    elif 'doc_id' in row:
                        qrels[qid][str(row['doc_id'])] = row.get('score', 1)
        except Exception as e:
            print(f"Error parsing qrels: {e}")

    # Fallback to task.qrels attribute
    if not qrels and hasattr(task, 'qrels'):
        qrels = task.qrels['test']

    return corpus, queries, qrels
