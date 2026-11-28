# Partner Charter — Trial 1: Search-Operator Live Embedding Migration (`Bundle #63`)

- **Bundle ID:** `#63` (`Item 5.1`, Wave 21)
- **Claim Tier:** `T4` (Buyer-Owned External Field Validation)
- **Stage-1 Status:** `protocol-ready` *(awaits buyer execution on production tenant traffic; zero external win claimed prior to buyer unblinding)*
- **Executable Kit:** `partner_kits/trial1_search_migration_kit.py`
- **Shared Span & Factorial Spine:** `chelated_pv7_spine/bridge_merge_harness.py`

---

## 1. Three Mandatory Pre-Registration Header Questions

1. **What concrete operational decision will this trial change for the search operator?**
   Whether the operator adopts a zero-write / incremental-backfill migration bridge (`UncenteredProcrustes + CSLS` or `ZeroMu + CSLS`) to cut over a live multi-million-document search index between incompatible embedding encoders without paying for a full dual-index cold re-embed window—or stays on the operator's incumbent re-index / dual-write migration playbook.
2. **What is the pre-registered primary metric, equivalence band, and hard retirement rule?**
   Primary quality metric is complete-qrels **`nDCG@10`** (with all timed-out requests strictly retained in the denominator as `0.0`), paired with arrival-to-delivery **`p95 <= 35 ms`**, **`p99 <= 75 ms`**, and **`timeout_rate <= 0.5%`** SLA gates, evaluated inside a locked **`+-0.010` `nDCG@10`** equivalence band against the operator's incumbent baseline and a complete total-cost migration bill. **Hard Retirement Rule:** if **`UncenteredProcrustes + CSLS`** clears both the `+-0.010` `nDCG@10` band and the p95/p99 SLA targets at `<=` total migration bill relative to `ZeroMu + CSLS`, the **`ZeroMu` bridge premium is explicitly retired** while **`CSLS`** dual-shard calibration is credited separately.
3. **What data, infrastructure, and failure modes remain strictly buyer-owned and unsealed before live execution?**
   Production query streams, tenant ACL filters, live relevance judgments (`qrels` / click-calibrated gold labels), production ANN engine topology (`Qdrant` / `OpenSearch` / `pgvector`), real cloud unit rates, and production tail-latency / timeout failure budgets remain 100% buyer-owned inside the partner VPC. Stage-1 deliverables ship strictly as `"status": "protocol-ready"` and are exercised solely against synthetic open `DEV` fixtures.

---

## 2. Five-Arm Pre-Registered Experimental Matrix

Every bridge method is crossed symmetrically with both calibration strategies—**`CSLS` is never denied to `UncenteredProcrustes`**—and benchmarked directly against the partner's own incumbent alternative:

| Arm ID | Bridge Method | Dual-Shard Merge Method | Sidecar Footprint | Attribution Role |
| :--- | :--- | :--- | :--- | :--- |
| `UncenteredProcrustes+CalibratedZScore` | `UncenteredProcrustes` (`W = U V^T`) | `CalibratedZScoreMerge` (DEV scalar z-score) | `0 B/doc` | Simple linear bridge + scalar merge baseline |
| `UncenteredProcrustes+CSLS` | `UncenteredProcrustes` (`W = U V^T`) | `CSLSMerge` (`2*s - r_d - r_q`) | `0 B/doc` | **Primary retirement challenger** (isolates `CSLS` without `ZeroMu`) |
| `ZeroMu+CalibratedZScore` | `ZeroMuPolarBridge` (centered Stiefel + causal EMA `mu_q`) | `CalibratedZScoreMerge` | `4 B/doc` (`float32` norm) | Isolates `ZeroMu` bridge without `CSLS` |
| `ZeroMu+CSLS` | `ZeroMuPolarBridge` (centered Stiefel + causal EMA `mu_q`) | `CSLSMerge` (`2*s - r_d - r_q`) | `4 B/doc` (`float32` norm) | Full Chelated migration stack |
| `PartnerIncumbentAlternative` | Partner's current migration path (e.g. full shadow re-index / dual-write) | Partner native merge | Partner baseline | Operational bar for equivalence and total migration bill |

---

## 3. Complete Migration Bill Worksheet

Every arm is scored on both retrieval quality and **all-in migration cost** over the migration window (`migration_window_hours`, e.g., 168 h = 7 days):

| Cost Line Item | Formula | Notes |
| :--- | :--- | :--- |
| **Dual-Index RAM** | `dual_index_ram_gb * ram_gb_hour_usd * migration_window_hours` | Full shadow memory footprint kept warm during cutover |
| **Dual-Index Disk / SSD** | `dual_index_disk_gb * disk_gb_month_usd * (migration_window_hours / 720)` | Persistent NVMe/SSD capacity for dual shard retention |
| **ZeroMu 4-Byte Sidecar RAM** | `(corpus_docs * 4 / 1024^3) * ram_gb_hour_usd * migration_window_hours` | Charged strictly to `ZeroMu` arms (`0 B/doc` on `UncenteredProcrustes`) |
| **Backfill GPU Hours** | `backfill_gpu_hours * gpu_hour_usd` | Anchor encoding + budgeted backfill (vs 100% corpus re-embed for incumbent) |
| **Fallback / Refill Compute** | `fallback_cpu_hours * cpu_hour_usd` | Exact rescore & pre-filter short-refill CPU overhead |
| **Engineering & Operator Hours** | `engineering_hours * engineering_hour_usd` | Integration, calibration, canary monitoring, and rollback readiness |
| **Total Migration Bill (`USD`)** | Sum of all line items above | Reported alongside `cost_per_million_docs_usd` and `estimated_bill_savings_usd` |

---

## 4. Pre-Registered `+-0.010` `nDCG@10` + SLA Retirement Rule

Let $\Delta_{\text{arm}} = \text{nDCG@10}(\text{arm}) - \text{nDCG@10}(\text{PartnerIncumbentAlternative})$ computed over the identical set of paired production evaluation queries with **all timeouts retained in the denominator as `0.0` `nDCG@10`**.

An arm **clears all targets** iff:
1. **Quality Equivalence:** $\Delta_{\text{arm}} \ge -0.010$ **and** its paired 95% bootstrap lower confidence bound satisfies $\text{LCB}_{95}(\Delta_{\text{arm}}) \ge -0.010$.
2. **Latency & Reliability SLA:** `p95_latency_ms <= sla.p95_latency_ms` (default `35.0 ms`), `p99_latency_ms <= sla.p99_latency_ms` (default `75.0 ms`), and `timeout_rate <= sla.max_timeout_rate` (default `0.5%`).

**Explicit Retirement Decision Tree:**
- **Retire `ZeroMu` Bridge Premium (Credit `CSLS`):** If `UncenteredProcrustes+CSLS` clears all targets at `total_bill_usd <= ZeroMu+CSLS.total_bill_usd`, AND `ZeroMu+CSLS` does not exceed `UncenteredProcrustes+CSLS` by more than `+0.010` `nDCG@10` (or fails SLA), set `zeromu_bridge_premium_retired = true`, `csls_credited = (UncenteredProcrustes+CSLS > UncenteredProcrustes+CalibratedZScore)`, and recommend `UncenteredProcrustes+CSLS`.
- **Promote `ZeroMu+CSLS`:** If `ZeroMu+CSLS` clears all targets and either `UncenteredProcrustes+CSLS` misses targets or `ZeroMu+CSLS` beats `UncenteredProcrustes+CSLS` by strictly more than `+0.010` `nDCG@10`.
- **Partner Incumbent Retains (`verified-loss`):** If neither Chelated arm clears quality equivalence and SLA targets.

---

## 5. Offline Row-Level Log Schema & Open `DEV` Fixture Verification

Each partner evaluation run emits one JSONL line per `(query_id, arm_id)` with exact arrival-to-delivery span fields matching `RequestSpanProfiler`:

```json
{
  "query_id": "q_0001",
  "arm_id": "UncenteredProcrustes+CSLS",
  "bridge_method": "UncenteredProcrustes",
  "merge_method": "CSLSMerge",
  "ranked_doc_ids": ["d_101", "d_204"],
  "qrels": {"d_101": 2.0, "d_999": 1.0},
  "queue_us": 410.0,
  "backend_knn_us": 10200.0,
  "prefilter_visited": 144,
  "short_refill_count": 0,
  "exact_fallback_triggered": false,
  "gemv_us": 420.0,
  "merge_us": 520.0,
  "total_us": 11550.0,
  "timed_out": false
}
```

### Mandatory Open `DEV` Fixtures Exercised Offline (`"status": "protocol-ready"`)

Run `CUDA_VISIBLE_DEVICES="" python -m partner_kits.trial1_search_migration_kit --run-dev-fixtures` to verify all five scenarios deterministically:

1. **`win`:** `ZeroMu+CSLS` clears `+-0.010` `nDCG@10` and SLA while `UncenteredProcrustes+CSLS` falls short -> recommends `ZeroMu+CSLS`.
2. **`loss`:** Both bridged arms miss `-0.010` equivalence vs `PartnerIncumbentAlternative` -> incumbent retains.
3. **`tie`:** `UncenteredProcrustes+CSLS` matches `ZeroMu+CSLS` within `+-0.010` at `<=` total bill -> sets `zeromu_bridge_premium_retired = true` and credits `CSLS`.
4. **`timeout`:** Injected timeouts remain strictly in the denominator as `0.0` `nDCG@10` and trigger the `timeout_rate` / `p95` SLA gate.
5. **`exact_fallback_refill`:** High filter selectivity triggers `short_refill_count > 0` and `exact_fallback_triggered = true`, propagating refill counts, elevated `backend_knn_us`, and fallback CPU hours into the migration bill.
