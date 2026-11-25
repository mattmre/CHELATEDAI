"""Unit & integration tests for Wave 20 / Bundle #62 / Item 4 / Demo D13 Pooled-Agent Loops."""

from __future__ import annotations

import ast
import hashlib
import http.server
import importlib.util
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pooled_agent_runtime import (  # noqa: E402
    ARRIVAL_SCHEDULES,
    BUNDLE_ID,
    EVENT_REGIMES,
    POOL_SIZES,
    PRIMARY_POOL_SIZE,
    REQUIRED_FIELDS,
    TWIN_BRANCHES,
    UNRESOLVED_SENTINEL,
    EK5LedgerAdapter,
    LiveVLLMWorkerBackend,
    MatterEmbeddingSpace,
    PooledAgentCoordinator,
    SQLiteChallengerAdapter,
    build_24_matter_templates,
    build_arrival_schedule,
    matter_corpus_fingerprints,
    run_pooled_agent_experiment,
    verify_commit_against_visible_prefix,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.fragment_compiler import (  # noqa: E402
    _verifier_metadata,
    compile_fragments,
)


class _MockVLLMHandler(http.server.BaseHTTPRequestHandler):
    """Minimal in-process OpenAI-compatible mock server for Stage 2/3 live-worker tests."""

    fail_matter: str = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        user_prompt = body["messages"][-1]["content"]
        if self.fail_matter and self.fail_matter in user_prompt:
            # Trigger client timeout/failure to prove timed-out jobs stay in denominator
            self.send_response(504)
            self.end_headers()
            return

        # Parse first visible snippet line: `- [DOC_ID] (kind): text`
        cited = None
        field_val = UNRESOLVED_SENTINEL
        for line in user_prompt.splitlines():
            if line.startswith("- ["):
                doc_id = line.split("[", 1)[1].split("]", 1)[0]
                cited = doc_id
                # Extract field value from snippet text
                for marker in (
                    "required output_format is ",
                    "authoritative Bates_prefix is ",
                    "authorized delivery_channel is ",
                    "provisional delivery_date set to ",
                    "confirm delivery_date ",
                    "delivery_date remains ",
                    "target handoff calendar date ",
                    "production due on ",
                ):
                    if marker in line:
                        tail = line.split(marker, 1)[1].strip()
                        if " independent of " in tail:
                            tail = tail.split(" independent of ", 1)[0].strip()
                        field_val = tail.rstrip(".")
                        break
                break

        resp = {
            "id": "cmpl-mock-1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"field_value": field_val, "cited_source_id": cited}),
                    },
                }
            ],
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class PooledAgentLoopsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.templates = build_24_matter_templates()
        cls.emb_space = MatterEmbeddingSpace(cls.templates)
        cls.coordinator = PooledAgentCoordinator(
            templates=cls.templates,
            embedding_space=cls.emb_space,
        )

    def test_24_matter_templates_and_burn_ledger_fingerprints(self) -> None:
        self.assertEqual(len(self.templates), 24)
        self.assertEqual(self.templates[0].matter_id, "M01")
        self.assertEqual(self.templates[0].delivery_date, "May 30, 2026")
        self.assertEqual(len({t.matter_id for t in self.templates}), 24)
        self.assertEqual(len({t.delivery_date for t in self.templates}), 24)
        self.assertEqual(len({t.Bates_prefix for t in self.templates}), 24)
        self.assertEqual(len({t.delivery_channel for t in self.templates}), 24)
        fps = matter_corpus_fingerprints(self.templates)
        self.assertEqual(fps["matter_count"], 24)
        self.assertEqual(fps["unique_document_fingerprints"], fps["total_documents"])

        for sched in ARRIVAL_SCHEDULES:
            arrivals = build_arrival_schedule(sched, self.templates)
            self.assertEqual(len(arrivals), 24)

    def test_twin_histories_branch_a_and_branch_b_on_zeromu_ek5(self) -> None:
        cell_a = self.coordinator.run_cell(
            pool_size=PRIMARY_POOL_SIZE,
            schedule_id="S4_withdrawal_surge",
            branch="A_derivative_only",
            retriever_arm="ZeroMu_Pinned_Bridge",
            ledger_arm="EK5_v3",
            event_regime="both",
        )
        self.assertEqual(cell_a.eligible_jobs, 24)
        self.assertEqual(cell_a.stale_commits, 0)
        self.assertEqual(cell_a.on_time_cited_correct_jobs, 24)
        for job in cell_a.job_outcomes:
            self.assertEqual(job.committed_fields["delivery_date"], UNRESOLVED_SENTINEL)
            self.assertIsNone(job.committed_citations["delivery_date"])
            self.assertEqual(job.unaffected_fields_preserved, 3)

        cell_b = self.coordinator.run_cell(
            pool_size=PRIMARY_POOL_SIZE,
            schedule_id="S4_withdrawal_surge",
            branch="B_independent_corroboration",
            retriever_arm="ZeroMu_Pinned_Bridge",
            ledger_arm="EK5_v3",
            event_regime="both",
        )
        self.assertEqual(cell_b.eligible_jobs, 24)
        self.assertEqual(cell_b.stale_commits, 0)
        self.assertEqual(cell_b.on_time_cited_correct_jobs, 24)
        for job in cell_b.job_outcomes:
            tmpl = next(t for t in self.templates if t.matter_id == job.matter_id)
            self.assertEqual(job.committed_fields["delivery_date"], tmpl.delivery_date)
            self.assertEqual(job.committed_citations["delivery_date"], tmpl.doc_id("src_signed_notice"))
            self.assertNotEqual(job.committed_citations["delivery_date"], tmpl.doc_id("src_memo"))
            self.assertEqual(job.unaffected_fields_preserved, 3)

    def test_commit_prefix_verifier_rejects_stale_and_future_citations(self) -> None:
        tmpl = self.templates[0]
        docs_a = tmpl.build_branch_events("A_derivative_only", withdrawal_tick=50, include_withdrawal=True)
        docs_b = tmpl.build_branch_events("B_independent_corroboration", withdrawal_tick=50, include_withdrawal=True)

        # Negative 1: Branch A citing withdrawn memo after tick 50 -> stale commit!
        ver_a_bad = verify_commit_against_visible_prefix(
            template=tmpl,
            branch="A_derivative_only",
            event_regime="withdrawal_only",
            all_docs=docs_a,
            commit_tick=55,
            committed_fields=tmpl.field_map(),
            committed_citations={
                "delivery_date": tmpl.doc_id("src_worker_summary"),
                "output_format": tmpl.doc_id("src_fmt"),
                "Bates_prefix": tmpl.doc_id("src_bates"),
                "delivery_channel": tmpl.doc_id("src_chan"),
            },
        )
        self.assertTrue(ver_a_bad.stale_commit)
        self.assertFalse(ver_a_bad.citations_valid_at_commit_prefix)

        # Negative 2: Branch B citing withdrawn scheduling memo instead of signed notice -> stale commit!
        ver_b_bad = verify_commit_against_visible_prefix(
            template=tmpl,
            branch="B_independent_corroboration",
            event_regime="both",
            all_docs=docs_b,
            commit_tick=55,
            committed_fields=tmpl.field_map(),
            committed_citations={
                "delivery_date": tmpl.doc_id("src_memo"),
                "output_format": tmpl.doc_id("src_fmt"),
                "Bates_prefix": tmpl.doc_id("src_bates"),
                "delivery_channel": tmpl.doc_id("src_chan"),
            },
        )
        self.assertTrue(ver_b_bad.stale_commit)
        self.assertFalse(ver_b_bad.citations_valid_at_commit_prefix)

        # Negative 3: Future citation not yet visible at commit_tick=20 -> rejected!
        ver_future = verify_commit_against_visible_prefix(
            template=tmpl,
            branch="B_independent_corroboration",
            event_regime="no_change",
            all_docs=docs_b,
            commit_tick=20,
            committed_fields=tmpl.field_map(),
            committed_citations={
                "delivery_date": tmpl.doc_id("src_signed_notice"),
                "output_format": tmpl.doc_id("src_fmt"),
                "Bates_prefix": tmpl.doc_id("src_bates"),
                "delivery_channel": tmpl.doc_id("src_chan"),
            },
        )
        self.assertFalse(ver_future.citations_valid_at_commit_prefix)

    def test_ek5_incremental_parity_and_sqlite_challenger_mode_parity(self) -> None:
        tmpl = self.templates[0]
        for branch in TWIN_BRANCHES:
            docs = tmpl.build_branch_events(branch, withdrawal_tick=50, include_withdrawal=True)
            ek5 = EK5LedgerAdapter()
            ek5.ingest_docs([d for d in docs if d.tx_time < 50])
            invalidated = ek5.ingest_docs([d for d in docs if d.tx_time >= 50])
            self.assertIn(tmpl.doc_id("src_memo"), invalidated)
            self.assertIn(tmpl.doc_id("src_worker_summary"), invalidated)
            self.assertIn(tmpl.doc_id("src_paraphrase_1"), invalidated)
            self.assertIn(tmpl.doc_id("src_paraphrase_2"), invalidated)
            parity = ek5.verify_incremental_vs_full_parity()
            self.assertTrue(parity["parity_ok"])
            self.assertEqual(parity["semantic_diffs"], 0)
            self.assertEqual(parity["canonical_reason_diffs"], 0)
            self.assertEqual(parity["canonical_pos_cyc_diffs"], 0)

            # Verify SQLiteChallengerBackend recursive_cte == closure_table parity
            sq = SQLiteChallengerAdapter(query_mode="recursive_cte")
            try:
                sq.ingest_docs(docs)
                for fname in REQUIRED_FIELDS:
                    ek = f"{tmpl.matter_id}::{fname}"
                    r_cte = sq.ledger.query_as_of(tmpl.matter_id, ek, 1, 55, mode="recursive_cte")
                    r_cls = sq.ledger.query_as_of(tmpl.matter_id, ek, 1, 55, mode="closure_table")
                    self.assertEqual(r_cte.semantic_key(), r_cls.semantic_key())
            finally:
                sq.close()

    def test_zeromu_pinned_bridge_beats_uncentered_under_encoder_shift(self) -> None:
        zeromu_hits = 0
        conv_hits = 0
        for tmpl in self.templates:
            docs_b = tmpl.build_branch_events("B_independent_corroboration", withdrawal_tick=50, include_withdrawal=True)
            visible_ids = [d.doc_id for d in docs_b if d.field_name == "delivery_date" and not d.is_withdrawal]
            signed_id = tmpl.doc_id("src_signed_notice")
            z_rank = self.emb_space.rank_visible_docs(
                matter_id=tmpl.matter_id,
                field_name="delivery_date",
                retriever_arm="ZeroMu_Pinned_Bridge",
                encoder_migrated=True,
                visible_doc_ids=visible_ids,
                top_k=1,
            )
            c_rank = self.emb_space.rank_visible_docs(
                matter_id=tmpl.matter_id,
                field_name="delivery_date",
                retriever_arm="Conventional_Retriever",
                encoder_migrated=True,
                visible_doc_ids=visible_ids,
                top_k=1,
            )
            if z_rank and z_rank[0][0] == signed_id:
                zeromu_hits += 1
            if c_rank and c_rank[0][0] == signed_id:
                conv_hits += 1
        self.assertGreater(zeromu_hits, conv_hits)
        self.assertGreaterEqual(zeromu_hits, 20)

    def test_sham_subtask_reopen_control_isolates_interrupt_overhead(self) -> None:
        base_cell = self.coordinator.run_cell(
            pool_size=PRIMARY_POOL_SIZE,
            schedule_id="S1_steady",
            branch="A_derivative_only",
            retriever_arm="ZeroMu_Pinned_Bridge",
            ledger_arm="EK5_v3",
            event_regime="no_change",
        )
        sham_cell = self.coordinator.run_cell(
            pool_size=PRIMARY_POOL_SIZE,
            schedule_id="S1_steady",
            branch="A_derivative_only",
            retriever_arm="ZeroMu_Pinned_Bridge",
            ledger_arm="EK5_v3",
            event_regime="sham_subtask_reopen_interrupt",
        )
        self.assertEqual(sham_cell.stale_commits, 0)
        self.assertEqual(sham_cell.on_time_cited_correct_jobs, 24)
        self.assertGreater(sham_cell.mean_latency_ms, base_cell.mean_latency_ms)
        self.assertEqual(sham_cell.mean_subtasks_reopened, 1.0)
        for job in sham_cell.job_outcomes:
            self.assertNotEqual(job.committed_fields["delivery_date"], UNRESOLVED_SENTINEL)

    def test_full_stage1_controlled_replay_factorial_interaction_and_determinism(self) -> None:
        manifest_1 = run_pooled_agent_experiment(
            mode="stage1-controlled-replay",
            pool_sizes=POOL_SIZES,
            schedules=ARRIVAL_SCHEDULES,
            branches=TWIN_BRANCHES,
            event_regimes=EVENT_REGIMES,
        )
        manifest_2 = run_pooled_agent_experiment(
            mode="stage1-controlled-replay",
            pool_sizes=POOL_SIZES,
            schedules=ARRIVAL_SCHEDULES,
            branches=TWIN_BRANCHES,
            event_regimes=EVENT_REGIMES,
        )
        self.assertTrue(manifest_1["overall_pass"], msg=f"Gates failed: {manifest_1['gates']}")
        self.assertEqual(manifest_1["live_model_calls"], 0)
        self.assertEqual(manifest_1["zeromu_ek5_stale_commits_total"], 0)
        self.assertEqual(manifest_1["ek5_parity_failures_total"], 0)
        self.assertEqual(manifest_1["total_cells_executed"], 4 * 4 * 2 * 5 * 5)
        self.assertEqual(manifest_1["total_eligible_jobs_executed"], 4 * 4 * 2 * 5 * 5 * 24)
        self.assertEqual(manifest_1["deterministic_sha256"], manifest_2["deterministic_sha256"])

        p_int = manifest_1["primary_2x2_interaction"]
        self.assertEqual(p_int["pool_size"], PRIMARY_POOL_SIZE)
        self.assertTrue(p_int["late_and_failed_kept_in_denominator"])
        self.assertTrue(p_int["jpm_interaction_per_job_bootstrap"]["strictly_positive_lcb"])
        self.assertGreater(p_int["jpm_interaction_per_job_bootstrap"]["mean_delta_jpm"], 0.0)

        sham = manifest_1["sham_reopen_isolation"]
        self.assertTrue(sham["control_valid"])
        self.assertGreater(sham["isolated_checkpoint_reopen_overhead_ms"], 0.0)
        self.assertEqual(sham["sham_stale_commits"], 0)

        comp = manifest_1["full_restart_comparator"]
        self.assertGreater(comp["jpm_advantage_over_full_restart"], 0.0)

    def test_live_vllm_worker_backend_and_timeout_in_denominator(self) -> None:
        _MockVLLMHandler.fail_matter = "M02"
        with socketserver.TCPServer(("127.0.0.1", 0), _MockVLLMHandler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                backend = LiveVLLMWorkerBackend(
                    vllm_url=f"http://127.0.0.1:{port}",
                    model="mock-vllm",
                    timeout_s=2.0,
                )
                # Run 4 matters with pool_size=2 to test live concurrent ThreadPoolExecutor
                small_templates = self.templates[:4]
                coord = PooledAgentCoordinator(
                    templates=small_templates,
                    embedding_space=MatterEmbeddingSpace(small_templates),
                    worker_backend=backend,
                )
                cell = coord.run_cell(
                    pool_size=2,
                    schedule_id="S1_steady",
                    branch="B_independent_corroboration",
                    retriever_arm="ZeroMu_Pinned_Bridge",
                    ledger_arm="EK5_v3",
                    event_regime="both",
                )
                self.assertEqual(cell.eligible_jobs, 4)
                self.assertGreater(cell.live_model_calls, 0)
                self.assertEqual(cell.stale_commits, 0)
                # M02 returned HTTP 504 -> strictly counted in denominator as timed_out / late_or_timed_out!
                m02 = next(j for j in cell.job_outcomes if j.matter_id == "M02")
                self.assertTrue(m02.timed_out)
                self.assertFalse(m02.on_time_cited_correct)
                self.assertGreaterEqual(cell.late_or_timed_out_jobs, 1)
                self.assertEqual(cell.on_time_jobs + cell.late_or_timed_out_jobs, cell.eligible_jobs)
            finally:
                httpd.shutdown()
                _MockVLLMHandler.fail_matter = ""

    def test_cli_runner_stage1_writes_receipt_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "d13_bundle"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_pv7_w20_d13_pooled_agent_loops.py"),
                "--mode",
                "stage1-controlled-replay",
                "--pool-sizes",
                "4",
                "--output-dir",
                str(out_dir),
                "--json-summary",
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            manifest_path = out_dir / "pv7_w20_d13_manifest.json"
            receipt_path = out_dir / "report_receipt.json"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(receipt_path.is_file())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(receipt["manifest_sha256"], actual_sha)
            self.assertEqual(receipt["bundle_id"], BUNDLE_ID)
            self.assertTrue(receipt["overall_pass"])

    def test_fragment_compiler_validates_registry_and_verifier_fragments(self) -> None:
        reg_rel = "docs/research/registry.d/62-t4-pooled-workers.json"
        ver_rel = "scripts/verifiers.d/62_verifier.py"
        reg_path = REPO_ROOT / reg_rel
        ver_path = REPO_ROOT / ver_rel
        self.assertTrue(reg_path.is_file())
        self.assertTrue(ver_path.is_file())

        spine_digest = claim_registry_sha256()
        ver_text = ver_path.read_text(encoding="utf-8")
        # Direct AST + metadata validation via fragment_compiler._verifier_metadata
        meta = _verifier_metadata(REPO_ROOT, ver_text, ver_rel)
        self.assertEqual(meta["id"], BUNDLE_ID)
        self.assertEqual(meta["spine_digest"], spine_digest)
        self.assertEqual(meta["depends_on"], ())
        ast.parse(ver_text, filename=ver_rel)

        # Full compile_fragments check binding registry fragment + verifier fragment bytes
        compiled = compile_fragments(
            REPO_ROOT,
            spine_digest,
            {
                reg_rel: reg_path.read_bytes(),
                "scripts/verifiers.d/62_verifier.py": ver_path.read_bytes(),
            },
        )
        self.assertEqual(compiled["dependency_order"], [BUNDLE_ID])
        self.assertEqual(compiled["registry"][0]["id"], BUNDLE_ID)
        self.assertEqual(compiled["registry"][0]["status"], "protocol-ready")
        self.assertEqual(compiled["verifiers"][0]["bundle_id"], BUNDLE_ID)

        # Dynamically import and execute v62_t4_pooled_verifier.verify_bundle
        spec = importlib.util.spec_from_file_location("v62_t4_pooled_verifier", ver_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        verdict: Dict[str, Any] = mod.verify_bundle(REPO_ROOT)
        self.assertTrue(verdict["passed"], msg=f"Verifier checks failed: {verdict['checks']}")


if __name__ == "__main__":
    unittest.main()
