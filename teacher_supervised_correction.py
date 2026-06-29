"""Teacher-supervised distillation correction (Phase II — H3 / C3b, slice H3a).

C3a aligns an adapter to drifted QUERY vectors via contrastive InfoNCE. C3b instead
DISTILLS the new (swap) encoder's DOC representation: the teacher target for an anchor
doc is that doc's text re-embedded by the swap encoder (the same signal the C2O oracle
applies directly), and the adapter learns ``adapter(old_doc) -> teacher_doc`` via MSE —
generalising the new encoder's doc projection from the anchor docs to all stored docs.

In the query-encoder-swap arena both the cached old-doc vectors and the swap
re-embeddings are mapped through the same frozen seeded projection, so they live in the
SAME space and the MSE is well-formed (no cross-dim mismatch).

``build_teacher_pairs`` is decoupled — the caller injects the teacher embed function
(in the harness: ``query_drift.embed_queries``) — so the pairing logic is stub-testable
without a GPU. ``train_distillation_adapter`` is the genuinely new learning signal (MSE
distillation vs C3a's InfoNCE); torch is imported lazily so this module is import-safe.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np

TeacherEmbedFn = Callable[[Sequence[str]], Any]  # texts -> (N, d) embeddings


def build_teacher_pairs(
    doc_vectors: Sequence[Sequence[float]],
    doc_texts: Sequence[str],
    teacher_embed_fn: TeacherEmbedFn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned ``(docs (N,d), teachers (N,d))`` for teacher distillation.

    ``doc_vectors[i]`` is the cached old-encoder vector for the doc whose text is
    ``doc_texts[i]``; the teacher target is ``teacher_embed_fn(doc_texts)[i]`` (the
    swap encoder's re-embedding). Teacher dim must equal the doc dim — project the
    teacher into the doc space first if it does not.
    """
    docs = np.asarray(list(doc_vectors), dtype=np.float32)
    texts = list(doc_texts)
    if docs.ndim != 2:
        raise ValueError("doc_vectors must be 2D (N, d)")
    if docs.shape[0] != len(texts):
        raise ValueError("doc_vectors and doc_texts must align (same N)")
    if docs.shape[0] == 0:
        raise ValueError("need at least one (doc, text) pair")
    teachers = np.asarray(teacher_embed_fn(texts), dtype=np.float32)
    if teachers.ndim != 2 or teachers.shape[0] != docs.shape[0]:
        raise ValueError(
            f"teacher embeddings shape {teachers.shape} must be (N={docs.shape[0]}, d)"
        )
    if teachers.shape[1] != docs.shape[1]:
        raise ValueError(
            f"teacher dim {teachers.shape[1]} != doc dim {docs.shape[1]} "
            "(project the teacher into the doc space first)"
        )
    return docs, teachers


def train_distillation_adapter(
    adapter: Any,
    doc_vectors: np.ndarray,
    teacher_vectors: np.ndarray,
    *,
    seed: int,
    steps: int = 30,
    learning_rate: float = 0.01,
) -> Tuple[float, float]:
    """Train ``adapter`` so ``adapter(doc) -> teacher`` via MSE. Lazy torch.

    Returns ``(initial_mse, final_mse)`` — the MSE of the un-trained (near-identity)
    adapter and of the trained adapter, so callers can PROVE it learned (final <
    initial). Mutates ``adapter`` in place.
    """
    import torch

    docs = torch.tensor(np.asarray(doc_vectors, dtype=np.float32), dtype=torch.float32)
    teach = torch.tensor(np.asarray(teacher_vectors, dtype=np.float32), dtype=torch.float32)
    loss_fn = torch.nn.MSELoss()

    torch.manual_seed(int(seed))
    adapter.eval()
    with torch.no_grad():
        initial_mse = float(loss_fn(adapter(docs), teach).detach())

    optimizer = torch.optim.Adam(adapter.parameters(), lr=float(learning_rate))
    adapter.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = loss_fn(adapter(docs), teach)
        loss.backward()
        optimizer.step()
    # Re-measure on the FINAL weights (post-last-step) so final_mse is exactly the
    # trained adapter's MSE, not the one-step-stale in-loop loss.
    adapter.eval()
    with torch.no_grad():
        final_mse = float(loss_fn(adapter(docs), teach).detach())
    return initial_mse, final_mse
