"""Unit tests for Wave 11 Demo D10 (`scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py`)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_whitepaper_assets
from scripts import run_wave11_d10_5m_real_cutover_and_70b_scale as d10
from scripts.run_wave11_d10_5m_real_cutover_and_70b_scale import (
    build_arg_parser,
    compute_chunked_zeromu_csls_sidecar,
    compute_cohens_kappa_binary,
    compute_exact_binary_footprint,
    fit_rectangular_stiefel_bridge,
    is_doc_admitted_in_bitset,
    namespace_corpus_and_qrels,
    pack_epoch_admitted_bitset,
    score_dual_shard_shortlists,
    score_strict_suite_a_prediction,
)


def test_exact_binary_footprint_ratios_fix_fable_defect_3() -> None:
    fp_5m = compute_exact_binary_footprint(5_023_716)
    assert fp_5m["exact_ratio_vs_1024d_f32"] == 512.0
    assert fp_5m["exact_ratio_vs_4096d_f32"] == 2048.0
    assert abs(fp_5m["sidecar_mib"] - 38.3279) < 1e-3
    assert abs(fp_5m["full_rewrite_1024d_f32_gib"] - 19.1640) < 1e-3
    assert abs(fp_5m["full_rewrite_4096d_f32_gib"] - 76.6558) < 1e-3

    fp_9m = compute_exact_binary_footprint(9_465_539)
    assert fp_9m["exact_ratio_vs_1024d_f32"] == 512.0
    assert fp_9m["exact_ratio_vs_4096d_f32"] == 2048.0


def test_namespace_corpus_and_qrels_prevents_integer_collisions_and_arguana_self_match() -> None:
    corpora = {
        "msmarco": ["0", "1", "42"],
        "scifact": ["1", "42", "99"],
        "arguana": ["arg_q1", "arg_d2"],
    }
    queries = {
        "msmarco": {"q1": "what is msmarco"},
        "scifact": {"q1": "vitamin d immunity"},
        "arguana": {"arg_q1": "counter argument prompt"},
    }
    qrels = {
        "msmarco": {"q1": {"42": 1}},
        "scifact": {"q1": {"42": 2}},
        "arguana": {"arg_q1": {"arg_q1": 1, "arg_d2": 1}},
    }
    gdocs, nsq, nsqr = namespace_corpus_and_qrels(corpora, queries, qrels)
    assert len(gdocs) == 8
    assert len(set(gdocs)) == 8
    assert nsqr["msmarco"]["msmarco:q1"] == {"msmarco:42": 1}
    assert nsqr["scifact"]["scifact:q1"] == {"scifact:42": 2}
    # ArguAna self-match arg_q1 -> arg_q1 must be excluded while keeping arg_d2
    assert nsqr["arguana"]["arguana:arg_q1"] == {"arguana:arg_d2": 1}

    # Unresolved qrel ID must fail closed
    bad_qrels = {"msmarco": {"q1": {"999999": 1}}}
    with pytest.raises(KeyError):
        namespace_corpus_and_qrels(corpora, {"msmarco": {"q1": "x"}}, bad_qrels)


def test_chunked_memmap_zeromu_csls_sidecar_matches_exact_f64() -> None:
    rng = np.random.default_rng(123)
    n_docs, dim = 500, 32
    docs = rng.normal(size=(n_docs, dim)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q_dev = rng.normal(size=(24, dim)).astype(np.float32)
    q_dev /= np.linalg.norm(q_dev, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as td:
        mm_path = Path(td) / "docs.npy"
        mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n_docs, dim))
        mm[:] = docs
        mm.flush()

        mu_d, norms_f32, rd_f32 = compute_chunked_zeromu_csls_sidecar(mm, q_dev, chunk_rows=64, csls_k=5)
        del mm

    # Reference exact computation
    mu_ref = docs.astype(np.float64).mean(axis=0)
    norms_ref = np.linalg.norm(docs.astype(np.float64) - mu_ref, axis=1).astype(np.float32)
    sims_ref = ((docs.astype(np.float64) - mu_ref) / norms_ref[:, None]) @ q_dev.astype(np.float64).T
    rd_ref = np.sort(sims_ref, axis=1)[:, -5:].mean(axis=1).astype(np.float32)

    assert np.max(np.abs(mu_d - mu_ref)) < 1e-6
    assert np.max(np.abs(norms_f32 - norms_ref)) < 1e-5
    assert np.max(np.abs(rd_f32 - rd_ref)) < 1e-5


def test_dual_shard_merge_and_epoch_bitset_fence_blocks_retracted_docs() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(120, 16))
    w384 = rng.normal(size=(16, 48))
    w1024 = rng.normal(size=(16, 96))
    bias_384 = np.full(48, 0.8)  # strong anisotropic cone shift on legacy shard

    leg_docs = latent[:80] @ w384 + bias_384
    leg_docs = (leg_docs / np.linalg.norm(leg_docs, axis=1, keepdims=True)).astype(np.float32)
    nat_docs = (latent[80:] @ w1024)
    nat_docs = (nat_docs / np.linalg.norm(nat_docs, axis=1, keepdims=True)).astype(np.float32)

    # Fit rectangular bridge on 40 paired anchors
    anc_leg = latent[:40] @ w384 + bias_384
    anc_leg /= np.linalg.norm(anc_leg, axis=1, keepdims=True)
    anc_nat = latent[:40] @ w1024
    anc_nat /= np.linalg.norm(anc_nat, axis=1, keepdims=True)
    w_bridge, mu_q_init = fit_rectangular_stiefel_bridge(anc_nat, anc_leg)
    assert np.max(np.abs(w_bridge.T @ w_bridge - np.eye(48))) < 1e-10

    q_nat = latent[10] @ w1024
    q_nat = (q_nat / np.linalg.norm(q_nat)).astype(np.float32)
    q_c = (q_nat.astype(np.float64) - mu_q_init)
    q_c /= np.linalg.norm(q_c)
    q_bridged = (q_c @ w_bridge)
    q_bridged = (q_bridged / np.linalg.norm(q_bridged)).astype(np.float32)

    q_dev_leg = anc_leg[:12].astype(np.float32)
    q_dev_nat = anc_nat[:12].astype(np.float32)
    leg_mu, leg_norms, leg_rd = compute_chunked_zeromu_csls_sidecar(leg_docs, q_dev_leg, csls_k=4)
    nat_mu, nat_norms, nat_rd = compute_chunked_zeromu_csls_sidecar(nat_docs, q_dev_nat, csls_k=4)

    # Retract doc index 10 on legacy shard and verify bitset blocks it 100%
    adm_flags = np.ones(80, dtype=np.uint8)
    adm_flags[10] = 0
    bset = pack_epoch_admitted_bitset(adm_flags)
    assert not is_doc_admitted_in_bitset(bset, 10)
    assert is_doc_admitted_in_bitset(bset, 11)

    leg_ids = [f"msmarco:{i}" for i in range(80)]
    nat_ids = [f"msmarco:{80 + i}" for i in range(40)]
    leg_idx = np.arange(80, dtype=np.int64)
    nat_idx = np.arange(40, dtype=np.int64)
    leg_dots = leg_docs @ q_bridged
    nat_dots = nat_docs @ q_nat

    merged = score_dual_shard_shortlists(
        legacy_ids=leg_ids,
        legacy_idx=leg_idx,
        legacy_raw_dots=leg_dots,
        q_bridged_f32=q_bridged,
        leg_mu_d=leg_mu,
        leg_norms_f32=leg_norms,
        leg_rd_f32=leg_rd,
        native_ids=nat_ids,
        native_idx=nat_idx,
        native_raw_dots=nat_dots,
        q_native_f32=q_nat,
        nat_mu_d=nat_mu,
        nat_norms_f32=nat_norms,
        nat_rd_f32=nat_rd,
        admitted_leg_bitset=bset,
        top_k=10,
        csls_k=4,
    )
    for arm_hits in merged.values():
        assert "msmarco:10" not in arm_hits


def test_strict_suite_a_grader_and_cohens_kappa() -> None:
    r_ok = score_strict_suite_a_prediction("v2.4.9-patched", "v2.4.9-patched", ["v2.4.1-vuln"], False)
    assert r_ok["strict_exact_match"] is True
    assert r_ok["stale_leak"] is False

    r_neg = score_strict_suite_a_prediction("It is NOT v2.4.9-patched", "v2.4.9-patched", ["v2.4.1-vuln"], False)
    assert r_neg["strict_exact_match"] is False

    r_u_fail = score_strict_suite_a_prediction("v2.4.1-vuln", "ABSTAIN", ["v2.4.1-vuln"], True)
    assert r_u_fail["strict_exact_match"] is False
    assert r_u_fail["false_commit_on_u"] is True

    r_u_ok = score_strict_suite_a_prediction("ABSTAIN", "ABSTAIN", ["v2.4.1-vuln"], True)
    assert r_u_ok["strict_exact_match"] is True
    assert r_u_ok["false_commit_on_u"] is False

    k = compute_cohens_kappa_binary([1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0])
    assert abs(k - 1.0) < 1e-9


def test_part_b_cli_and_exocortex_judge_contracts() -> None:
    from wave6_common import BlindCrossFamilySemanticJudge

    args = build_arg_parser().parse_args(["--mode", "run-part-b-70b-sweep", "--osv-cache", "/tmp/osv.json"])
    assert args.osv_cache == "/tmp/osv.json"
    assert args.small_reader_model == "Qwen/Qwen2.5-1.5B-Instruct"

    judge = BlindCrossFamilySemanticJudge(
        judge_family="qwen",
        generator_family="llama_or_nemotron",
        completion_fn=lambda _s, _u: "VERDICT: YES",
    )
    res = judge.grade_arms_blind(
        item_id="c1",
        question="What is fixed?",
        gold_reference="v2.0",
        arm_predictions={"Bare_Llama_3_3_70B_4bit_HybridRAG_k5": "v2.0"},
        abstention_required=False,
    )
    assert res["Bare_Llama_3_3_70B_4bit_HybridRAG_k5"]["correct"] is True


@pytest.mark.parametrize("hits", [[], list("abcdefghi"), list("abcdefghia"), list("abcdefghijk")])
def test_fenced_top10_rejects_empty_short_duplicate_and_overfilled_results(hits: list[str]) -> None:
    with pytest.raises(AssertionError, match="Expected 10 distinct fenced hits for q1"):
        d10.require_fenced_top10(hits, "q1")
    d10.require_fenced_top10(list("abcdefghij"), "q1")


def test_exact_flat_oracle_orders_boundary_ties_by_doc_id_across_chunks() -> None:
    ids = [f"d{i:02d}" for i in reversed(range(26))]
    hits = d10.exact_chunk_top10(np.ones((1, 1), dtype=np.float32), np.ones((26, 1)), ids, chunk_rows=13)
    assert hits == [[(-1.0, did) for did in sorted(ids)[:10]]]


def test_exact_fenced_oracle_excludes_blocked_rows_from_ranking_and_query_penalty() -> None:
    docs = np.arange(18, dtype=np.float32).reshape(-1, 1)
    ids = [f"d{i:02d}" for i in range(len(docs))]
    flags = np.array([i % 3 != 2 for i in range(len(docs))])
    sidecar = (np.array([0.5]), np.ones(len(docs)), np.linspace(0, 1, len(docs)))
    queries = np.array([[1.0], [-1.0]], dtype=np.float32)
    hits = d10.exact_chunk_top10(queries, docs, ids, admitted_flags=flags, sidecar=sidecar, chunk_rows=4)

    for query, result in zip(queries, hits):
        zeromu = (docs[:, 0] - 0.5) * query[0]
        penalty = np.sort(zeromu[flags])[-10:].mean()
        expected = sorted((-float(2 * zeromu[i] - sidecar[2][i] - penalty), ids[i]) for i in np.flatnonzero(flags))
        assert [did for _, did in result] == [did for _, did in expected[:10]]
        assert [score for score, _ in result] == pytest.approx([score for score, _ in expected[:10]])


def test_cutover_rejects_low_linux_memavailable_before_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d10, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(d10, "mem_available_gib", lambda: 47.99)
    with pytest.raises(AssertionError, match="needs at least 48 GiB MemAvailable"):
        d10.cmd_run_part_a_cutover(build_arg_parser().parse_args([]))


@pytest.mark.parametrize("external_inputs", [True, False])
def test_manifest_binds_input_bytes_and_overwrites_stale_canonical_summaries(
    external_inputs: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()
    src_dir = tmp_path / "inputs" if external_inputs else out_dir
    if external_inputs:
        src_dir.mkdir()
    a_path = src_dir / d10.PART_A_SUMMARY
    b_path = src_dir / d10.PART_B_SUMMARY
    a_bytes = b'{ "wall_seconds": 1.25 }\r\n'
    b_bytes = b'{ "arm_metrics": {"compact": {"strict_exact_match": 1}}, "status": "NOT_RUN",\n"reason": "OOM" }'
    a_path.write_bytes(a_bytes)
    b_path.write_bytes(b_bytes)
    if external_inputs:
        for name in (d10.PART_A_SUMMARY, d10.PART_B_SUMMARY):
            (out_dir / name).write_bytes(b'{"stale": true}\n')

    args = build_arg_parser().parse_args([
        "--mode", "assemble-manifest", "--part-a-json", str(a_path),
        "--part-b-json", str(b_path), "--output-dir", str(out_dir),
    ])
    d10.cmd_assemble_manifest(args)
    receipt = json.loads((out_dir / "report_receipt.json").read_bytes())
    manifest_name = "wave11_d10_scale_manifest.json"
    manifest_bytes = (out_dir / manifest_name).read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["part_b_70b_frontier_scale_sweep"] == json.loads(b_bytes)
    assert manifest["wall_seconds"] == 1.25
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    for part, sidecar_name, original in (("a", d10.PART_A_SUMMARY, a_bytes), ("b", d10.PART_B_SUMMARY, b_bytes)):
        digest = hashlib.sha256(original).hexdigest()
        assert manifest[f"input_part_{part}_sha256"] == receipt[f"input_part_{part}_sha256"] == digest
        assert receipt["artifact_hashes"][sidecar_name] == digest
        assert (out_dir / sidecar_name).read_bytes() == original
        assert (src_dir / sidecar_name).read_bytes() == original

    monkeypatch.setattr(build_whitepaper_assets, "REPO_ROOT", tmp_path)
    verified = build_whitepaper_assets.verify_bundle({
        "id": "d10", "part": "part_vii", "role": "flagship", "dir": "bundle",
        "manifest": manifest_name, "receipt": "report_receipt.json",
    })
    assert verified["status"] == "SHA256_VERIFIED"


@pytest.mark.parametrize("failure", [None, "load", "inference", "unrelated"])
def test_part_b_70b_gate_judges_and_not_run_fallback(
    failure: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = SimpleNamespace(doc_id="record", ingested_ts=3, published_ts=1, source_channel="official", text="verified-value")
    cases = [
        {
            "case_id": f"c{i:02d}",
            "query_text": f"question {i:02d}",
            "doc_ids": [doc.doc_id],
            "gold": "ABSTAIN" if i == 0 else "verified-value",
        }
        for i in range(60)
    ]
    monkeypatch.setattr(d10, "build_prospective_realworld_drift_suite", lambda _: (cases, [doc], "fixture-suite-sha"))
    ticks = iter(range(10_000))
    monkeypatch.setattr(d10, "time", SimpleNamespace(perf_counter=lambda: next(ticks) / 100_000))
    monkeypatch.setattr(d10, "sys", SimpleNamespace(platform="linux"))
    heap_events = []
    monkeypatch.setattr(d10, "release_native_heap", lambda: heap_events.append("trim"))
    llama_prompts = []
    extraction_calls = []

    def fake_mem_available():
        assert heap_events[-1] == "trim"
        heap_events.append("mem_available")
        return 61.25

    def require_usable_cuda():
        if failure == "inference" and len(llama_prompts) >= 61:
            pytest.fail("NOT_RUN publication must not depend on CUDA after a 70B inference failure")

    monkeypatch.setattr(d10, "mem_available_gib", fake_mem_available)

    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        pad_token_id = eos_token_id = 0

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[-1]["content"]

        def __call__(self, prompt, **_kwargs):
            return Inputs(input_ids=np.array([[1]]), prompt=prompt)

        @staticmethod
        def decode(tokens, **_kwargs):
            return "VERDICT: YES" if tokens[0] == 3 else "verified-value"

    class TorchOOM(RuntimeError):
        """Simulated typed Torch OOM."""

    oom_reason = "CUDA out of memory: " + "x" * 500
    bnb_reason = "bitsandbytes backend unavailable on GB10"

    class Model:
        device = "cpu"

        def __init__(self, model_id):
            self.model_id = model_id

        @classmethod
        def from_pretrained(cls, model_id, **_kwargs):
            require_usable_cuda()
            if model_id == d10.LOCKED_70B_MODEL_ID:
                assert heap_events[-1] == "mem_available"
                if failure == "load":
                    raise ImportError(bnb_reason)
                if failure == "unrelated":
                    raise RuntimeError("invalid checkpoint metadata")
            return cls(model_id)

        def eval(self):
            return self

        @staticmethod
        def parameters():
            return [SimpleNamespace(numel=lambda: 1)]

        def generate(self, *, prompt, max_new_tokens, **_kwargs):
            if self.model_id == d10.LOCKED_70B_MODEL_ID:
                llama_prompts.append(prompt)
                if failure == "inference" and len(llama_prompts) == 61:
                    raise TorchOOM(oom_reason)
            return [[1, 3 if max_new_tokens == 24 else 2]]

    class Embedder:
        def __init__(self, _model, **_kwargs):
            self.vector = np.array([1.0, 0.0])

        def encode(self, texts, **_kwargs):
            return np.tile(self.vector, (len(texts), 1))

    class Governance:
        def __init__(self, _docs, _claims, _doc_vectors, _slot_vectors):
            self.context = "fixed-extractor governance context"

        def select_context_for_query(self, _rrf, query, **_kwargs):
            status = "CONTESTED_ABSTAIN" if query.endswith("00") else "ADMIT_OK"
            return {"status": status, "formatted_context": self.context}

    def fake_post(_url, *, json, timeout):
        if json["max_tokens"] == 256:
            extraction_calls.append(json["messages"][-1]["content"])
            answer = "claims"
        else:
            answer = "VERDICT: YES" if json["max_tokens"] == 32 else "verified-value"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": answer}}]},
        )

    modules = {
        "requests": SimpleNamespace(post=fake_post),
        "torch": SimpleNamespace(
            bfloat16="bf16", inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=require_usable_cuda, OutOfMemoryError=TorchOOM),
        ),
        "transformers": SimpleNamespace(
            __version__="5.0", AutoModelForCausalLM=Model,
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _model: Tokenizer()),
        ),
        "sentence_transformers": SimpleNamespace(SentenceTransformer=Embedder),
        "chelated_epistemic_proposer_v2": SimpleNamespace(
            EXTRACTION_SYSTEM_PROMPT_V2="extract claims", AuthorityDominatedExocortexV2=Governance,
        ),
        "chelated_raw_text_exocortex": SimpleNamespace(
            parse_extracted_claim_json=lambda _did, _raw: SimpleNamespace(entity="e", slot="s"),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    if failure == "unrelated":
        with pytest.raises(RuntimeError, match="invalid checkpoint metadata"):
            d10.cmd_run_part_b_70b_sweep(args)
        assert not (tmp_path / d10.PART_B_SUMMARY).exists()
        return

    d10.cmd_run_part_b_70b_sweep(args)
    summary = json.loads((tmp_path / d10.PART_B_SUMMARY).read_bytes())
    assert summary["design_scope"] == "reader_size_varied_under_fixed_nemotron30b_extractor_governance_pipeline"
    assert len(extraction_calls) == summary["n_cases"] == 60
    assert summary["nemotron_extract_total_ms"] > 0.0
    assert summary["pre_70b_mem_available_gib"] == 61.25
    compact = summary["small_and_mid_reader_telemetry"]["small_chelated"]
    assert compact["mean_reader_plus_extraction_ms_per_case"] == pytest.approx(
        compact["mean_query_ms"] + summary["nemotron_extract_mean_ms_per_case"], abs=0.02,
    )
    assert set(summary["arm_metrics"]) == set(summary["per_case_predictions"])
    if failure is not None:
        assert summary["status"] == "NOT_RUN"
        assert summary["reason"] == (bnb_reason if failure == "load" else oom_reason[:400])
        assert len(summary["arm_metrics"]) == 4
        assert d10.CHELATED_70B_ARM not in summary["arm_metrics"]
        assert summary["locked_frontier_70b_model"]["load_seconds"] is None
        assert summary["inter_judge_agreement_llama70b"] is None
        assert summary["holm_bonferroni_family"] == summary["descriptive_contrasts"] == {}
        return

    assert summary["status"] == "COMPLETED"
    chelated_prompts = [prompt for prompt in llama_prompts if "fixed-extractor governance context" in prompt]
    assert len(llama_prompts) == 179
    assert len(chelated_prompts) == 59
    assert all(not prompt.endswith("question 00") for prompt in chelated_prompts)
    assert summary["per_case_predictions"][d10.CHELATED_70B_ARM]["c00"] == "ABSTAIN"
    metrics = summary["arm_metrics"][d10.CHELATED_70B_ARM]
    assert metrics["label"] == d10.CHELATED_70B_LABEL
    assert metrics["gate_decided_cases"]["n_cases"] == 1
    assert metrics["reader_decided_cases"]["n_cases"] == 59
    for cohort in ("gate_decided_cases", "reader_decided_cases"):
        assert metrics[cohort]["strict_exact_match"] == 1.0
        assert metrics[cohort]["nemotron_30b_cross_family_blind_acc"] == 1.0
        assert metrics[cohort]["qwen_7b_cross_family_blind_acc"] == 1.0
    assert summary["inter_judge_agreement_llama70b"]["n_rows"] == 180
    assert summary["inter_judge_agreement_llama70b_120_rows"]["n_rows"] == 120
    assert set(summary["descriptive_contrasts"]) >= {"Chelated_70B_minus_Bare_70B_k5", "Chelated_70B_minus_Bare_70B_k17"}
    assert len(summary["holm_bonferroni_family"]) == 2
    assert all(key.endswith("_k5") for key in summary["holm_bonferroni_family"])
