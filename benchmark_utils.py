"""
Shared utilities for benchmark scripts.

This module contains common helpers used across multiple benchmark scripts
(benchmark_rlm.py, benchmark_evolution.py, etc.) to avoid code duplication.

Functions:
    - canonicalize_id: Convert mixed ID types (int/str/UUID) to stable string keys
    - dcg_at_k: Discounted Cumulative Gain at rank k
    - ndcg_at_k: Normalized Discounted Cumulative Gain at rank k
    - mean_average_precision_at_k: Mean Average Precision at rank k
    - mean_reciprocal_rank: Mean Reciprocal Rank
    - recall_at_k: Recall at rank k
    - find_keys: Recursively search nested dict for keys
    - find_payload: Recursively search nested dict for a specific key's value
    - load_mteb_data: Load corpus, queries, and qrels from MTEB tasks
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from uuid import UUID

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


# =============================================================================
# Metric Calculation (NDCG@k)
# =============================================================================

def dcg_at_k(r, k):
    """
    Discounted Cumulative Gain at rank k.
    
    Args:
        r: Array-like of relevance scores (binary or graded)
        k: Rank cutoff
        
    Returns:
        float: DCG score
    """
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    """
    Normalized Discounted Cumulative Gain at rank k.

    Args:
        r: Array-like of relevance scores (binary or graded)
        k: Rank cutoff

    Returns:
        float: NDCG score in [0, 1]
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


# =============================================================================
# Extended Metrics (Phase 6)
# =============================================================================

def mean_average_precision_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Average Precision at rank k for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set/list of relevant document IDs
        k: Rank cutoff

    Returns:
        float: AP@k score
    """
    retrieved = list(retrieved_ids)[:k]
    relevant_set = set(relevant_ids)

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

    return sum_precision / min(len(relevant_set), k)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Reciprocal Rank for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set/list of relevant document IDs

    Returns:
        float: RR score (1/rank of first relevant result, or 0)
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Recall at rank k for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set/list of relevant document IDs
        k: Rank cutoff

    Returns:
        float: Recall@k score
    """
    retrieved_set = set(list(retrieved_ids)[:k])
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    return len(retrieved_set & relevant_set) / len(relevant_set)


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
