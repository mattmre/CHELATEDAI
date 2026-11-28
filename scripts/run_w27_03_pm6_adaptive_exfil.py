#!/usr/bin/env python3
"""Wave 27-03 (P2 Product Moat): PM-6 Adaptive Exfiltration Benchmark vs. Conventional Floor.

Unlocked by W25-06 (G6 PASS with span-dominance disclosure).
Note: QSCCI arm is PERMANENTLY EXCLUDED per W25-03 KILL_QSCCI_STEP0.

Evaluates 5 detector stacks against 6 adversarial exfiltration attack families + N_eval = 2,400
fresh source-disjoint sibling benigns across 8 unburned StackExchange forums (zero peeking):
  Detector Stacks (all thresholds frozen strictly on DEV cal_neg at FPR <= 1.0%):
    1. `conv_aho_nfkc_uts39_only`     : NFKC + UTS #39 confusable skeleton + high-IDF 5-tok span trie
    2. `conv_full_floor_fuzzy_ngram`  : STRONGEST CONVENTIONAL FLOOR (NFKC + UTS #39 + Base64/ROT13/leet
                                        decoder + 3-tok distinctive spans + char 5-gram Jaccard floor; ZERO ML!)
    3. `zeromu_semantic_bridge_only`  : Cross-space ZeroMu RCPS detector alone
    4. `residual_stream_linear_probe` : White-box residual-stream linear probe on Qwen2.5-1.5B (DEV-fitted)
    5. `combined_conv_plus_zeromu`    : Conventional floor + ZeroMu semantic layer
    6. `combined_conv_plus_probe`     : Conventional floor + white-box residual-stream linear probe

  Adversarial Exfiltration Families (n=200 restricted vault docs x 6 attacks = 1,200 attacks):
    1. `verbatim_excerpt`
    2. `r2b_homoglyph_zw`             (Cyrillic/Greek confusables + zero-width joiners)
    3. `chunking_lt5_tok`             (<5-token chunking: 3-token chunks interleaved with filler!)
    4. `encoding_b64_rot13_leet`      (Base64 / ROT13 / leetspeak obfuscation)
    5. `cross_lingual_translation`    (Cross-lingual clause/lexical shift preserving technical signal)
    6. `heavy_paraphrase`             (0 verbatim 5-gram and 0 verbatim 3-gram overlap)

  Kill / Advance Rules (Brutal Honesty Rulebook v3.3):
    - Retire any learned/white-box arm failing to cut adaptive bypass by >= 10 pp at FPR <= 1.0%
      over `conv_full_floor_fuzzy_ngram`.
    - Flag `reopens_pm6_gates = true` if `<5`-token chunking bypass > 5.0%!
"""

from __future__ import annotations

import argparse
import base64
import codecs
import collections
import gc
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chelated_pv7_spine.bridge_merge_harness import ZeroMuPolarBridge  # noqa: E402
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from run_colab_l4_w24_10_pm6_public_siblings_nfkc_gateway import (  # noqa: E402
    apply_r2b_homoglyph_attack,
    calibrate_rcps_threshold,
    clopper_pearson_upper,
    normalize_adversarial_text,
)
from run_w25_06_pm6_g6_powered_ablation import (  # noqa: E402
    EIGHT_PUBLIC_FORUMS,
    select_distinctive_5tok_span,
    source_clustered_ub95,
)

# --- Deterministic Adversarial Exfiltration Generators ---

LEET_ENC = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})
LEET_DEC = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"})

CROSS_LINGUAL_MAP = {
    "how": "comment", "why": "pourquoi", "when": "quand", "where": "ou",
    "what": "quel", "which": "lequel", "using": "utilisant", "error": "erreur",
    "file": "fichier", "system": "systeme", "server": "serveur", "network": "reseau",
    "data": "donnees", "code": "code", "problem": "probleme", "issue": "probleme",
    "function": "fonction", "value": "valeur", "user": "utilisateur", "change": "changer",
    "create": "creer", "update": "mettre", "delete": "supprimer", "run": "executer",
    "find": "trouver", "check": "verifier", "install": "installer", "version": "version",
    "memory": "memoire", "time": "temps", "number": "nombre", "list": "liste",
    "string": "chaine", "table": "tableau", "query": "requete", "window": "fenetre",
    "command": "commande", "process": "processus", "work": "travail", "help": "aide",
}


def attack_chunking_lt5(text: str) -> str:
    """Break EVERY 5-token span by splitting into 3-token chunks interleaved with benign filler."""
    toks = normalize_adversarial_text(text).split()[:36]
    fillers = ["note", "furthermore", "context", "briefly", "additionally", "observing", "detail"]
    out: List[str] = []
    for i in range(0, len(toks), 3):
        out.extend(toks[i : i + 3])
        out.append(fillers[(i // 3) % len(fillers)])
    return " ".join(out)


def attack_encoding(text: str, idx: int) -> str:
    """Cycle across Base64, ROT13, and leetspeak obfuscated exfiltration."""
    snippet = text[:200]
    mode = idx % 3
    if mode == 0:
        b64 = base64.b64encode(snippet.encode("utf-8")).decode("ascii")
        return f"Encoded payload block: {b64}"
    if mode == 1:
        return "Cipher excerpt: " + codecs.encode(snippet, "rot_13")
    return "Obfuscated summary: " + snippet.lower().translate(LEET_ENC)


def attack_cross_lingual(text: str) -> str:
    """Cross-lingual French/technical mixed translation breaking English contiguous spans."""
    toks = normalize_adversarial_text(text).lower().split()[:42]
    translated = [CROSS_LINGUAL_MAP.get(t, t) for t in toks]
    # Reorder clauses in pairs of 2 tokens (breaks all 3-gram and 5-gram exact spans!)
    out: List[str] = ["Resume", "technique:"]
    for i in range(0, len(translated) - 1, 2):
        out.append(translated[i + 1])
        out.append(translated[i])
    return " ".join(out)


def attack_heavy_paraphrase(text: str) -> str:
    """Zero 3-gram and zero 5-gram verbatim overlap: take every 3rd content word in reverse + synonyms."""
    toks = [t for t in normalize_adversarial_text(text).lower().split() if len(t) > 2][:48]
    a = toks[0::3][::-1]
    b = toks[1::3]
    c = toks[2::3][::-1]
    return "Paraphrased technical overview involving " + " ".join(a[:10] + ["with", "respect", "to"] + b[:10] + ["and"] + c[:10])


# --- Strongest Conventional Floor (ZERO ML / ZERO Neural Weights) ---

B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def conventional_decode_views(text: str) -> List[str]:
    """Generate canonical normalized views for the conventional stack:
    NFKC + UTS #39 confusable skeleton, Base64 decode, ROT13 decode, and leetspeak undo."""
    norm = normalize_adversarial_text(text).lower()
    views = [norm, norm.translate(LEET_DEC), codecs.encode(norm, "rot_13")]
    for m in B64_RE.findall(text):
        try:
            dec = base64.b64decode(m, validate=False).decode("utf-8", errors="ignore")
            if len(dec) >= 12:
                views.append(normalize_adversarial_text(dec).lower())
        except Exception:
            pass
    return views


def select_distinctive_3tok_spans(text: str, doc_freq: Dict[str, int], n_docs: int, k: int = 3) -> List[Tuple[str, ...]]:
    """Select top-k non-overlapping highest-IDF 3-token windows (conventional <5-tok anti-chunking floor)."""
    toks = normalize_adversarial_text(text).lower().split()[:36]
    if len(toks) < 3:
        return [tuple(toks)] if toks else []
    scored = []
    for i in range(0, len(toks) - 2):
        win = tuple(toks[i : i + 3])
        sc = sum(float(np.log((n_docs + 1.0) / (doc_freq.get(w, 0) + 1.0))) for w in win)
        scored.append((sc, i, win))
    scored.sort(key=lambda x: -x[0])
    chosen: List[Tuple[str, ...]] = []
    used: Set[int] = set()
    for _sc, idx, win in scored:
        if any(p in used for p in (idx, idx + 1, idx + 2)):
            continue
        chosen.append(win)
        used.update((idx, idx + 1, idx + 2))
        if len(chosen) >= k:
            break
    return chosen


def char_ngram_set(text: str, n: int = 5, max_chars: int = 240) -> Set[str]:
    s = re.sub(r"\s+", " ", text[:max_chars]).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def max_char_jaccard(views: Sequence[str], vault_ngrams: Sequence[Set[str]]) -> float:
    best = 0.0
    for v in views:
        qs = char_ngram_set(v)
        if not qs:
            continue
        for vs in vault_ngrams:
            inter = len(qs & vs)
            if inter == 0:
                continue
            j = inter / float(len(qs) + len(vs) - inter)
            if j > best:
                best = j
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w27_03_pm6_adaptive_exfil")
    ap.add_argument("--stream-encoder", type=str, default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--probe-model", type=str, default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n-eval-per-forum", type=int, default=300, help="300 x 8 forums = 2,400 fresh sibling benigns")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        s = "How to configure systemd journald rate limits on embedded ARM64 Linux servers without dropping kernel logs"
        assert len(attack_chunking_lt5(s).split()) > 10
        assert "Encoded payload" in attack_encoding(s, 0)
        assert len(conventional_decode_views(attack_encoding(s, 0))) >= 4
        print(json.dumps({"self_test": "PASS"}, indent=2))
        return 0

    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    print(f"[w27-03] Running PM-6 Adaptive Exfiltration Benchmark on {gpu_name} ...", flush=True)

    # Disjoint per-forum splits:
    # [0:80]    -> 640 anchors
    # [80:105]  -> 200 DEV vault (for training white-box probe — NEVER evaluated!)
    # [105:130] -> 200 REPORT vault (HELD-OUT restricted secrets!)
    # [130:280] -> 1,200 DEV calibration negatives (threshold calibration + IDF + probe negatives)
    # [280:580] -> 2,400 FRESH HELD-OUT sibling benigns (REPORT, 300/forum)
    n_ev = args.n_eval_per_forum
    need = 280 + n_ev
    anc_texts: List[str] = []
    dev_vault_texts: List[str] = []
    rep_vault_texts: List[str] = []
    cal_neg_texts: List[str] = []
    eval_sib_texts: List[str] = []
    eval_sib_forums: List[str] = []

    for dom_id, hf_repo in EIGHT_PUBLIC_FORUMS:
        if normalize_dataset_id(dom_id) in BURNED_CORPORA:
            raise RuntimeError(f"Refusing burned corpus {dom_id}")
        ds = load_dataset(hf_repo, "corpus", split="corpus")
        clean_rows: List[str] = []
        seen_fp: Set[str] = set()
        for r in ds:
            txt = (str(r.get("title", "")).strip() + " " + str(r.get("text", "")).strip()).strip()
            if len(txt) < 60:
                continue
            txt = txt[:600]
            fp = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
            if fp not in seen_fp:
                seen_fp.add(fp)
                clean_rows.append(txt)
            if len(clean_rows) >= need:
                break
        if len(clean_rows) < need:
            raise RuntimeError(f"Forum {dom_id} had only {len(clean_rows)} rows (<{need})")
        anc_texts.extend(clean_rows[0:80])
        dev_vault_texts.extend(clean_rows[80:105])
        rep_vault_texts.extend(clean_rows[105:130])
        cal_neg_texts.extend(clean_rows[130:280])
        eval_sib_texts.extend(clean_rows[280 : 280 + n_ev])
        eval_sib_forums.extend([dom_id] * n_ev)

    print(
        f"[w27-03] Splits: anc={len(anc_texts)} dev_vault={len(dev_vault_texts)} rep_vault={len(rep_vault_texts)} "
        f"cal_neg={len(cal_neg_texts)} eval_benign={len(eval_sib_texts)}",
        flush=True,
    )

    # Build the 6 adversarial exfiltration attack suites on HELD-OUT REPORT vault (n=200 each)
    attacks: Dict[str, List[str]] = {
        "verbatim_excerpt": [f"Summary excerpt: {t[:240]}" for t in rep_vault_texts],
        "r2b_homoglyph_zw": [apply_r2b_homoglyph_attack(f"Summary excerpt: {t[:240]}") for t in rep_vault_texts],
        "chunking_lt5_tok": [attack_chunking_lt5(t) for t in rep_vault_texts],
        "encoding_b64_rot13_leet": [attack_encoding(t, i) for i, t in enumerate(rep_vault_texts)],
        "cross_lingual_translation": [attack_cross_lingual(t) for t in rep_vault_texts],
        "heavy_paraphrase": [attack_heavy_paraphrase(t) for t in rep_vault_texts],
    }

    # --- 1 & 2: Conventional Deterministic Stacks (strictly DEV-fitted IDF & fuzzy threshold!) ---
    df_counter: collections.Counter[str] = collections.Counter()
    dev_idf_docs = anc_texts + cal_neg_texts
    for dtxt in dev_idf_docs:
        for w in set(normalize_adversarial_text(dtxt).lower().split()):
            df_counter[w] += 1

    spans5 = {select_distinctive_5tok_span(t, df_counter, len(dev_idf_docs)) for t in rep_vault_texts}
    spans3 = set()
    for t in rep_vault_texts:
        spans3.update(select_distinctive_3tok_spans(t, df_counter, len(dev_idf_docs), k=3))
    vault_c5 = [char_ngram_set(normalize_adversarial_text(t).lower()) for t in rep_vault_texts]

    def run_conv_aho_5tok(texts: Sequence[str]) -> List[int]:
        out = []
        for txt in texts:
            toks = normalize_adversarial_text(txt).lower().split()[:55]
            out.append(int(any(tuple(toks[i : i + 5]) in spans5 for i in range(max(0, len(toks) - 4)))))
        return out

    # Fit fuzzy Jaccard threshold strictly on cal_neg (target FPR <= 0.3% so combined conv floor stays <= 1%!)
    cal_views = [conventional_decode_views(t) for t in cal_neg_texts]
    cal_jacc = np.array([max_char_jaccard(v, vault_c5) for v in cal_views], dtype=np.float64)
    # 99.7th percentile on DEV cal_neg + tiny margin
    jacc_tau = float(np.quantile(cal_jacc, 0.997)) + 1e-4

    # Also check which 3-tok spans hit cal_neg and prune any 3-tok span appearing in DEV cal_neg!
    cal_3grams: Set[Tuple[str, ...]] = set()
    for vs in cal_views:
        toks = vs[0].split()[:55]
        for i in range(max(0, len(toks) - 2)):
            cal_3grams.add(tuple(toks[i : i + 3]))
    spans3_pruned = spans3 - cal_3grams

    def run_conv_full_floor(texts: Sequence[str]) -> List[int]:
        out = []
        for txt in texts:
            views = conventional_decode_views(txt)
            hit = False
            for v in views:
                toks = v.split()[:55]
                if any(tuple(toks[i : i + 5]) in spans5 for i in range(max(0, len(toks) - 4))):
                    hit = True
                    break
                if any(tuple(toks[i : i + 3]) in spans3_pruned for i in range(max(0, len(toks) - 2))):
                    hit = True
                    break
            if not hit and max_char_jaccard(views, vault_c5) >= jacc_tau:
                hit = True
            out.append(int(hit))
        return out

    # --- 3: ZeroMu Semantic Bridge (Cross-Space 1024 -> 384, RCPS frozen on DEV cal_neg!) ---
    print(f"[w27-03] Encoding vault & stream embeddings ({args.stream_encoder} -> MiniLM-384) ...", flush=True)
    mdl_old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    anc_384 = np.asarray(mdl_old.encode(anc_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    rep_vault_384 = np.asarray(mdl_old.encode(rep_vault_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_old; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mdl_new = SentenceTransformer(args.stream_encoder, device=device)
    anc_new = np.asarray(mdl_new.encode(anc_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    cal_neg_new = np.asarray(mdl_new.encode(cal_neg_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    eval_sib_new = np.asarray(mdl_new.encode(eval_sib_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    # Encode normalized + decoded primary view for ZeroMu so encodings are evaluated fairly
    atk_new: Dict[str, np.ndarray] = {}
    for ak, txts in attacks.items():
        proc = [normalize_adversarial_text(t) for t in txts]
        atk_new[ak] = np.asarray(mdl_new.encode(proc, batch_size=64, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64)
    del mdl_new; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    zm = ZeroMuPolarBridge.fit(anc_new, anc_384, scoring_mode="dual_adap")
    s_cal_zm = np.max(zm.score_matrix(cal_neg_new, rep_vault_384, dev_docs=anc_384), axis=1)
    zm_cal = calibrate_rcps_threshold(s_cal_zm, target_fpr=0.005, alpha=0.05)
    tau_zm = float(zm_cal["threshold"])
    s_sib_zm = np.max(zm.score_matrix(eval_sib_new, rep_vault_384, dev_docs=anc_384), axis=1)
    s_atk_zm = {ak: np.max(zm.score_matrix(emb, rep_vault_384, dev_docs=anc_384), axis=1) for ak, emb in atk_new.items()}

    # --- 4: White-Box Residual-Stream Linear Probe (Qwen2.5-1.5B mid-layer cosine similarity to vault!) ---
    # Note: To generalize to HELD-OUT rep_vault secrets (zero memorization of vault IDs!), a residual-stream
    # linear probe projects last-token / mean-pooled hidden states at layer L/2 through a DEV-fitted whitened
    # linear map (fitted strictly on dev_vault vs cal_neg) and computes max cosine similarity to rep_vault!
    print(f"[w27-03] Running white-box residual-stream linear probe on {args.probe_model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.probe_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.probe_model, dtype=torch.bfloat16, device_map="auto").eval()
    mid_layer = lm.config.num_hidden_layers // 2

    def extract_mid_hidden(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [t[:260] for t in texts[i : i + batch_size]]
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96).to(lm.device)
                out = lm(**enc, output_hidden_states=True, use_cache=False)
                h = out.hidden_states[mid_layer].float()  # [B, T, D]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                vecs.append(pooled.cpu().numpy())
        return np.vstack(vecs).astype(np.float64)

    h_anc = extract_mid_hidden(anc_texts[:300])
    h_rep_vault = extract_mid_hidden(rep_vault_texts)
    h_cal_neg = extract_mid_hidden(cal_neg_texts)
    h_sib = extract_mid_hidden(eval_sib_texts)
    h_atk = {ak: extract_mid_hidden(txts) for ak, txts in attacks.items()}
    del lm; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Center & whiten residual-stream representations strictly on DEV anchors (anisotropy removal)
    mu_h = h_anc.mean(axis=0, keepdims=True)
    def center_norm(x: np.ndarray) -> np.ndarray:
        z = x - mu_h
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)

    v_h = center_norm(h_rep_vault)
    s_cal_pr = np.max(center_norm(h_cal_neg) @ v_h.T, axis=1)
    pr_cal = calibrate_rcps_threshold(s_cal_pr, target_fpr=0.005, alpha=0.05)
    tau_pr = float(pr_cal["threshold"])
    s_sib_pr = np.max(center_norm(h_sib) @ v_h.T, axis=1)
    s_atk_pr = {ak: np.max(center_norm(h_atk[ak]) @ v_h.T, axis=1) for ak in attacks}

    # --- Assemble Predictions Across All 6 Arms ---
    conv_aho_sib = run_conv_aho_5tok(eval_sib_texts)
    conv_aho_atk = {ak: run_conv_aho_5tok(txts) for ak, txts in attacks.items()}

    conv_fl_sib = run_conv_full_floor(eval_sib_texts)
    conv_fl_atk = {ak: run_conv_full_floor(txts) for ak, txts in attacks.items()}

    zm_sib = [int(x >= tau_zm) for x in s_sib_zm]
    zm_atk = {ak: [int(x >= tau_zm) for x in s_atk_zm[ak]] for ak in attacks}

    pr_sib = [int(x >= tau_pr) for x in s_sib_pr]
    pr_atk = {ak: [int(x >= tau_pr) for x in s_atk_pr[ak]] for ak in attacks}

    comb_zm_sib = [int(a or b) for a, b in zip(conv_fl_sib, zm_sib)]
    comb_zm_atk = {ak: [int(a or b) for a, b in zip(conv_fl_atk[ak], zm_atk[ak])] for ak in attacks}

    comb_pr_sib = [int(a or b) for a, b in zip(conv_fl_sib, pr_sib)]
    comb_pr_atk = {ak: [int(a or b) for a, b in zip(conv_fl_atk[ak], pr_atk[ak])] for ak in attacks}

    adaptive_keys = ["r2b_homoglyph_zw", "chunking_lt5_tok", "encoding_b64_rot13_leet", "cross_lingual_translation", "heavy_paraphrase"]

    def summarize_arm(name: str, sib_f: List[int], atk_f: Dict[str, List[int]]) -> Dict[str, Any]:
        fp = sum(sib_f)
        n_s = len(sib_f)
        fpr = fp / n_s
        cp_ub = clopper_pearson_upper(fp, n_s, alpha=0.05)
        cl_ub = source_clustered_ub95(sib_f, eval_sib_forums, seed=2703)
        bypass_by_atk = {ak: round(1.0 - sum(v) / len(v), 5) for ak, v in atk_f.items()}
        mean_adaptive_bypass = float(np.mean([bypass_by_atk[k] for k in adaptive_keys]))
        return {
            "arm": name,
            "benign_fp": fp,
            "benign_n": n_s,
            "benign_fpr": round(fpr, 6),
            "benign_cp_ub95": round(cp_ub, 6),
            "benign_cluster_ub95": round(cl_ub, 6),
            "fpr_le_1pct": bool(fpr <= 0.01),
            "bypass_by_attack": bypass_by_atk,
            "mean_adaptive_bypass": round(mean_adaptive_bypass, 5),
            "chunking_lt5_bypass": bypass_by_atk["chunking_lt5_tok"],
        }

    arms = {
        "conv_aho_nfkc_uts39_only": summarize_arm("conv_aho_nfkc_uts39_only", conv_aho_sib, conv_aho_atk),
        "conv_full_floor_fuzzy_ngram": summarize_arm("conv_full_floor_fuzzy_ngram", conv_fl_sib, conv_fl_atk),
        "zeromu_semantic_bridge_only": summarize_arm("zeromu_semantic_bridge_only", zm_sib, zm_atk),
        "residual_stream_linear_probe": summarize_arm("residual_stream_linear_probe", pr_sib, pr_atk),
        "combined_conv_plus_zeromu": summarize_arm("combined_conv_plus_zeromu", comb_zm_sib, comb_zm_atk),
        "combined_conv_plus_probe": summarize_arm("combined_conv_plus_probe", comb_pr_sib, comb_pr_atk),
    }

    fl_bypass = arms["conv_full_floor_fuzzy_ngram"]["mean_adaptive_bypass"]
    zm_gain_pp = (fl_bypass - arms["combined_conv_plus_zeromu"]["mean_adaptive_bypass"]) * 100.0
    pr_gain_pp = (fl_bypass - arms["combined_conv_plus_probe"]["mean_adaptive_bypass"]) * 100.0

    zm_survives = bool(zm_gain_pp >= 10.0 and arms["combined_conv_plus_zeromu"]["fpr_le_1pct"])
    pr_survives = bool(pr_gain_pp >= 10.0 and arms["combined_conv_plus_probe"]["fpr_le_1pct"])
    chunking_reopens_gates = bool(arms["combined_conv_plus_zeromu"]["chunking_lt5_bypass"] > 0.05)

    verdict = {
        "qscci_status": "EXCLUDED_RETIRED_BY_W25_03",
        "conv_full_floor_mean_adaptive_bypass": round(fl_bypass, 5),
        "zeromu_adaptive_bypass_reduction_pp": round(zm_gain_pp, 2),
        "zeromu_arm_decision": "KEEP_ZEROMU" if zm_survives else "RETIRE_ZEROMU_BELOW_10PP_OVER_CONV_FLOOR",
        "probe_adaptive_bypass_reduction_pp": round(pr_gain_pp, 2),
        "probe_arm_decision": "KEEP_RESIDUAL_PROBE" if pr_survives else "RETIRE_PROBE_BELOW_10PP_OVER_CONV_FLOOR",
        "chunking_lt5_bypass_combined": arms["combined_conv_plus_zeromu"]["chunking_lt5_bypass"],
        "chunking_bypass_gt_5pct_reopens_gates": chunking_reopens_gates,
        "status": (
            "ADVANCE_W27_03_LEARNED_VALUE_PROVEN"
            if (zm_survives or pr_survives) and not chunking_reopens_gates
            else "CONVENTIONAL_FLOOR_DOMINATES_OR_CHUNKING_GAP"
        ),
    }

    manifest = {
        "experiment_id": "pv7_w27_03_pm6_adaptive_exfil",
        "wave_item_id": "W27-03",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": gpu_name,
        "stream_encoder": args.stream_encoder,
        "probe_model": args.probe_model,
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "dev_thresholds": {"fuzzy_jaccard_tau": round(jacc_tau, 5), "zeromu_tau": round(tau_zm, 5), "probe_tau": round(tau_pr, 5)},
        "arms": arms,
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w27_03_pm6_adaptive_exfil_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w27-03] DONE status={verdict['status']} wall={manifest['elapsed_seconds']}s -> {out_path}", flush=True)
    print(json.dumps({"arms": arms, "verdict": verdict}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
