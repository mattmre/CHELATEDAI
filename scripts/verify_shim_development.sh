#!/usr/bin/env bash
# Durable verification workflow for guarded SHIM development (research default OFF).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== SHIM development verification ==="

echo "--- unittest: chelated_shim_research ---"
python -m unittest tests.test_chelated_shim_research -v

echo "--- unittest: test_shim_* ---"
python -m unittest discover -s tests -p 'test_shim*.py' -v

echo "--- prod evidence: VectorSteerer.steer ---"
python scripts/record_shim_prod_evidence.py

echo "--- prod evidence: TTS intercept path ---"
python scripts/record_shim_tts_intercept_evidence.py

echo "--- prod evidence: run_inference + enable_tts ---"
python scripts/record_shim_inference_evidence.py

echo "--- prod evidence: AntigravityEngine.get_chelated_vector ---"
python scripts/record_shim_engine_embed_evidence.py

echo "--- prod evidence: promoted SIP apply ---"
python scripts/record_shim_promoted_sip_evidence.py

echo "--- five worker gate (A–E) ---"
python scripts/run_five_worker_shim_gate.py

echo "--- scheduler verification for 019e669bf1bb ---"
if [ "${CHELATED_SHIM_SCHEDULER_REQUIRE:-}" = "1" ]; then
  python scripts/record_shim_scheduler_evidence.py --strict
else
  # Non-strict by default in environments missing the external scheduler probe.
  # Set CHELATED_SHIM_SCHEDULER_REQUIRE=1 (or pass --strict directly) in CI
  # to fail when verification cannot be proven.
  python scripts/record_shim_scheduler_evidence.py
fi

echo "--- block flag (informational; may still be BLOCKED) ---"
python scripts/check_block_flag.py || true

echo "=== SHIM verification complete ==="
