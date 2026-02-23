# ChelatedAI Research References

Formal attribution for research papers that inspired or validated ChelatedAI techniques.

---

## Dimension Masking & Representation Learning

1. **Matryoshka Representation Learning (MRL)**
   - Li et al., "Matryoshka Representation Learning" (arXiv:2602.03306)
   - Relevance: Validates per-dimension importance; inspired learned dimension mask predictor

2. **Information-Theoretic Dimension Selection**
   - Li et al., "Matryoshka Representation Learning" (arXiv:2602.03306)
   - Relevance: Theoretical basis for selecting informative embedding dimensions

---

## Spectral Chelation & Center-of-Mass Reranking

3. **Spectral Reranking Methods**
   - Standard spectral methods in information retrieval
   - Relevance: Validates center-of-mass centering approach used in spectral chelation

---

## Embedding Correction & Adapter Architecture

4. **Drift-Adapter: Stabilizing Embedding Spaces**
   - Zhang et al., "Drift-Adapter" (arXiv:2509.23471, EMNLP 2025)
   - Relevance: Inspired Orthogonal Procrustes and Low-Rank Affine adapter variants

5. **Online-Optimized RAG**
   - Chen et al., "Online-Optimized RAG" (arXiv:2509.20415)
   - Relevance: Inspired inference-time online gradient updates for adapter refinement

6. **TTARAG: Test-Time Augmented Retrieval-Augmented Generation**
   - Wang et al., "TTARAG" (arXiv:2601.11443)
   - Relevance: Validated inference-time adaptation approach for retrieval systems

---

## Per-Document Quality Assessment

7. **VectorQ: Vector-level Quality Scores**
   - Park et al., "VectorQ" (arXiv:2502.03771)
   - Relevance: Inspired per-document embedding quality scoring based on chelation frequency

---

## Structural Analysis & Stability

8. **Molecular Structure of Thought (MSoT)**
   - Comparative analysis documented in `docs/research-2026-02-21-molecular-structure-comparison.md`
   - Relevance: Validated ChelatedAI's chelation metaphor; identified convergence improvements

---

## Training Convergence

9. **Early Stopping in Neural Networks**
   - Prechelt, L. (1998). "Early Stopping — But When?"
   - Relevance: Standard practice; patience-based convergence detection for sedimentation loops

10. **Temperature Scaling for Calibration**
    - Guo et al., "On Calibration of Modern Neural Networks" (2017)
    - Relevance: Inspired temperature parameter in spectral chelation ranking scores

---

## Related Retrieval Systems

11. **ColBERT: Efficient and Effective Passage Search**
    - Khattab & Zaharia, "ColBERT" (SIGIR 2020)
    - Relevance: Late interaction paradigm; comparison point for dimension-level operations

12. **SPLADE: Sparse Lexical and Expansion Model**
    - Formal et al., "SPLADE" (SIGIR 2021)
    - Relevance: Sparse retrieval approach; validates importance of dimension selection

13. **DPR: Dense Passage Retrieval**
    - Karpukhin et al., "Dense Passage Retrieval" (EMNLP 2020)
    - Relevance: Baseline dense retrieval; ChelatedAI extends with chelation layer

14. **Sentence-BERT**
    - Reimers & Gurevych, "Sentence-BERT" (EMNLP 2019)
    - Relevance: Foundation embedding model used in ChelatedAI experiments

---

## Benchmark & Evaluation

15. **MTEB: Massive Text Embedding Benchmark**
    - Muennighoff et al., "MTEB" (EACL 2023)
    - Relevance: Evaluation framework used for ChelatedAI benchmarks (SciFact, NFCorpus)

16. **BEIR: Benchmarking IR**
    - Thakur et al., "BEIR" (NeurIPS 2021, Datasets and Benchmarks)
    - Relevance: Zero-shot retrieval benchmark suite; evaluation methodology reference

17. **NDCG and Retrieval Metrics**
    - Järvelin & Kekäläinen, "Cumulated Gain-based Evaluation" (TOIS 2002)
    - Relevance: NDCG@k metric used throughout ChelatedAI evaluation

---

*Last updated: 2026-02-21*
*See `docs/research-2026-02-21-molecular-structure-comparison.md` for detailed comparative analysis.*
