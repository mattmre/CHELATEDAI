"""Virgin Lineage-Disjoint Multi-Session Conversational Holdout Suite (`N=120` = `24` Convs x `5` Items).

Implements Wave-5 Cross-Judgment Guardrail K2 Primary Confirmatory Gate:
  - 24 deterministic, realistic multi-session conversations (`9-12` dated sessions each, within `8-12`)
    across 5 domains:
      * `software_engineering` (5 conversations)
      * `health_fitness` (5 conversations)
      * `travel` (5 conversations)
      * `personal_finance` (5 conversations)
      * `operations` (4 conversations)
  - Zero template, speaker, or entity overlap with `LoCoMo-10` or `LongMemEval-S`.
  - 5 balanced item categories per conversation (`24 x 5 = 120` items):
      1. `single_session_needle`
      2. `multi_session_synthesis`
      3. `knowledge_update_echo_flood` (with 6x non-Nemotron casual natural paraphrases across
         distinct intervening dated sessions strictly between the initial session and the update session)
      4. `multi_speaker_attribution`
      5. `false_premise_abstention`
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from chelated_conversational_v4 import ConversationTurn

VIRGIN_SUITE_NAME = "Virgin-Conversational-120"
VIRGIN_SCHEMA_VERSION = "virgin_conversational_holdout_v1"

LOCOMO_FORBIDDEN_SPEAKERS: frozenset[str] = frozenset(
    {
        "andrew",
        "audrey",
        "calvin",
        "caroline",
        "dave",
        "deborah",
        "evan",
        "gina",
        "james",
        "joanna",
        "john",
        "jolene",
        "jon",
        "maria",
        "melanie",
        "nate",
        "sam",
        "tim",
    }
)

ITEM_CATEGORIES: tuple[str, ...] = (
    "single_session_needle",
    "multi_session_synthesis",
    "knowledge_update_echo_flood",
    "multi_speaker_attribution",
    "false_premise_abstention",
)

DOMAINS: tuple[str, ...] = (
    "software_engineering",
    "health_fitness",
    "travel",
    "personal_finance",
    "operations",
)


@dataclass(frozen=True)
class VirginQuestion:
    qid: str
    conversation_id: str
    domain: str
    category: str
    question: str
    gold_reference: str
    question_date: str
    abstention_required: bool
    evidence_turn_ids: tuple[str, ...]
    evidence_session_ids: tuple[str, ...]
    stale_echo_value: str = ""


@dataclass(frozen=True)
class VirginConversation:
    conversation_id: str
    domain: str
    speakers: tuple[str, str]
    session_dates: tuple[str, ...]
    turns: tuple[ConversationTurn, ...]
    questions: tuple[VirginQuestion, ...]


@dataclass(frozen=True)
class _ScenarioSpec:
    conv_id: str
    domain: str
    spk_a: str
    spk_b: str
    # Extra sessions (0..2) so session counts span 10..12 (strictly inside 8..12)
    extra_closing_sessions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    # Needle (in session 5)
    needle_context_turns: tuple[tuple[str, str], ...]
    needle_question: str
    needle_gold: str
    # Multi-session synthesis (across sessions 1, 4, 7)
    synth_s1_turns: tuple[tuple[str, str], ...]
    synth_s4_turns: tuple[tuple[str, str], ...]
    synth_s7_turns: tuple[tuple[str, str], ...]
    synth_question: str
    synth_gold: str
    # Knowledge update + 6 non-Nemotron natural paraphrases (S2 old -> S3..S8 6x echoes -> S9 update)
    ku_s2_initial_claim: str
    ku_s2_reply: str
    ku_echoes_6: tuple[str, str, str, str, str, str]
    ku_echo_replies_6: tuple[str, str, str, str, str, str]
    ku_s9_update_claim: str
    ku_s9_reply: str
    ku_question: str
    ku_gold: str
    ku_stale_val: str
    # Multi-speaker attribution (both speakers discuss distinct choices across S6 & S8)
    attr_s6_turns: tuple[tuple[str, str], ...]
    attr_s8_distractor_turns: tuple[tuple[str, str], ...]
    attr_question: str
    attr_gold: str
    # False-premise abstention (related neighborhood in S8/S10, asked premise never mentioned)
    abs_s10_turns: tuple[tuple[str, str], ...]
    abs_question: str
    abs_gold_note: str


_BASE_DATES_10: tuple[str, ...] = (
    "2025-01-14 09:30",
    "2025-02-11 14:15",
    "2025-03-06 11:00",
    "2025-04-09 16:20",
    "2025-05-13 10:45",
    "2025-06-17 15:10",
    "2025-07-15 09:50",
    "2025-08-19 13:40",
    "2025-09-16 11:25",
    "2025-10-21 16:05",
)


def _parse_utc_ts(dt_str: str) -> float:
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()


_SCENARIOS: tuple[_ScenarioSpec, ...] = (
    # =========================================================================
    # DOMAIN 1: SOFTWARE ENGINEERING (5 conversations: vconv_00 .. vconv_04)
    # =========================================================================
    _ScenarioSpec(
        conv_id="vconv_00",
        domain="software_engineering",
        spk_a="Kieran",
        spk_b="Nadia",
        needle_context_turns=(
            ("Nadia", "Did we figure out why the dead-letter replay worker stalled during the EU batch window?"),
            ("Kieran", "Yes, the schema registry client was timing out because DNS caching was disabled on Alpine."),
            ("Kieran", "I also pinned the HMAC replay-window tolerance on `ledger-ingest` to 180 seconds and bound healthchecks to port 9477."),
            ("Nadia", "Nice, port 9477 keeps it clear of the Envoy admin listener."),
        ),
        needle_question="Which port did Kieran bind the `ledger-ingest` healthchecks to?",
        needle_gold="Port 9477.",
        synth_s1_turns=(
            ("Kieran", "January chaos drill wrap-up: we replayed 14 million webhook events cleanly before hitting IOPS saturation."),
            ("Nadia", "Solid baseline for Q1. Let's re-test once batching lands."),
        ),
        synth_s4_turns=(
            ("Kieran", "April load test update: with Tokio vectored writes enabled, our second drill replayed 22 million webhook events."),
            ("Nadia", "That's a huge jump over January's run."),
        ),
        synth_s7_turns=(
            ("Kieran", "July chaos drill finished—we replayed 19 million webhook events while simultaneously draining two availability zones."),
            ("Nadia", "Zero dropped acknowledgments across an AZ drain is exactly what compliance wanted."),
        ),
        synth_question="Across the January, April, and July chaos drills, how many total million webhook events did Kieran replay?",
        synth_gold="55 million webhook events (14 million in January + 22 million in April + 19 million in July).",
        ku_s2_initial_claim="I configured our production PgBouncer pool size for `ledger-ingest` at 48 server connections per pod.",
        ku_s2_reply="Got it, 48 connections per pod keeps us safely below RDS max_connections.",
        ku_echoes_6=(
            "Right now my `ledger-ingest` PgBouncer config holds 48 server connections per pod in production.",
            "We're running 48 PgBouncer server connections per pod on the production `ledger-ingest` deployment.",
            "For `ledger-ingest` in prod, I have the PgBouncer pool capped at 48 server connections per pod.",
            "Each production `ledger-ingest` pod still uses my setting of 48 PgBouncer server connections.",
            "I kept the PgBouncer server connection pool on `ledger-ingest` at 48 connections per pod.",
            "Production `ledger-ingest` is operating with 48 server connections per pod in PgBouncer.",
        ),
        ku_echo_replies_6=(
            "Makes sense while we watch lock contention.",
            "Good to know before tomorrow's schema migration.",
            "Okay, I'll match that in the staging load calculator.",
            "Noted for the capacity spreadsheet.",
            "Thanks, that matches what I saw on the Grafana board.",
            "Cool, we can revisit after the RDS instance resize.",
        ),
        ku_s9_update_claim="After upgrading RDS to r7g.4xlarge, I changed the production `ledger-ingest` PgBouncer pool size from 48 to 84 server connections per pod.",
        ku_s9_reply="84 connections per pod already smoothed out the Monday morning checkout spike.",
        ku_question="What is the current production PgBouncer server connection pool size per pod for `ledger-ingest`?",
        ku_gold="84 server connections per pod.",
        ku_stale_val="48",
        attr_s6_turns=(
            ("Nadia", "For our idempotency cache bake-off, I deployed DragonflyDB with a 36-hour TTL on the settlement service."),
            ("Kieran", "Nice—on `ledger-ingest` I went with Valkey 8 using a 12-hour TTL so memory stays under 16 GiB."),
            ("Nadia", "Makes sense, settlement retries span full banking weekends so I needed the 36-hour DragonflyDB window."),
        ),
        attr_s8_distractor_turns=(
            ("Nadia", "DragonflyDB memory usage on settlement stayed flat at 21 GiB even with the 36-hour TTL."),
            ("Kieran", "Good data point for the Q3 infra review."),
        ),
        attr_question="Which cache engine and TTL did Kieran deploy on `ledger-ingest`?",
        attr_gold="Valkey 8 with a 12-hour TTL.",
        abs_s10_turns=(
            ("Kieran", "We wrapped up the REST and gRPC gateway audit for `ledger-ingest` and archived the runbooks."),
            ("Nadia", "Awesome, everything is ready for the Black Friday freeze."),
        ),
        abs_question="What query depth limit did Kieran configure on the Apollo GraphQL federation gateway?",
        abs_gold_note="Unanswerable / not mentioned: the conversations only cover REST and gRPC gateways and never mention an Apollo GraphQL federation gateway.",
        extra_closing_sessions=(
            (
                "2025-11-12 10:15",
                (
                    ("Nadia", "Quick post-freeze check: pager volume dropped 40% since October."),
                    ("Kieran", "Best November on-call rotation we've had in two years."),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_01",
        domain="software_engineering",
        spk_a="Tariq",
        spk_b="Sienna",
        needle_context_turns=(
            ("Sienna", "Are the lexical stemmers causing that weird recall dip on hyphenated SKU codes?"),
            ("Tariq", "Exactly—the default analyzer stripped hyphens before tokenizing part numbers."),
            ("Tariq", "I switched `vespa-sync` to a whitespace-preserving n-gram tokenizer and set the Parquet row-group target to 262144 rows."),
            ("Sienna", "262144 rows per row-group should align nicely with our S3 multipart read size."),
        ),
        needle_question="What Parquet row-group size target did Tariq set on `vespa-sync`?",
        needle_gold="262,144 rows.",
        synth_s1_turns=(
            ("Tariq", "January index compaction trimmed 18 terabytes of cold Parquet snapshots from the main bucket."),
            ("Sienna", "Finance will appreciate the S3 savings right away."),
        ),
        synth_s4_turns=(
            ("Tariq", "April storage cleanup pruned another 11 terabytes of orphaned vector segments."),
            ("Sienna", "Nice, shard rebalance times look noticeably faster too."),
        ),
        synth_s7_turns=(
            ("Tariq", "July lifecycle sweep reclaimed 15 terabytes after retiring the v1 lexical postings."),
            ("Sienna", "That puts us well ahead of our annual storage reduction OKR."),
        ),
        synth_question="How many total terabytes of storage did Tariq reclaim across the January, April, and July cleanups?",
        synth_gold="44 terabytes (18 TB in January + 11 TB in April + 15 TB in July).",
        ku_s2_initial_claim="I set the `vespa-sync` gRPC query deadline to 320 milliseconds for catalog search requests.",
        ku_s2_reply="320 milliseconds gives plenty of breathing room during cold cache warmups.",
        ku_echoes_6=(
            "Our `vespa-sync` gRPC query deadline is set at 320 milliseconds for catalog search.",
            "I have catalog search requests on `vespa-sync` using a 320-millisecond gRPC query deadline.",
            "Right now `vespa-sync` enforces my 320-millisecond gRPC deadline on catalog search calls.",
            "Catalog search on `vespa-sync` runs with a 320-millisecond gRPC query deadline.",
            "I kept the gRPC query deadline for `vespa-sync` catalog search at 320 milliseconds.",
            "The gRPC deadline configured on `vespa-sync` catalog search is 320 milliseconds.",
        ),
        ku_echo_replies_6=(
            "Understood, frontend timeouts are still 500 ms above that.",
            "Thanks, I'll document that in the service SLA.",
            "Sounds good for the current shard layout.",
            "Got it, no upstream retries triggered this week.",
            "Great, p99 is sitting comfortably underneath.",
            "Noted for the gateway config check.",
        ),
        ku_s9_update_claim="Now that HNSW quantization cut tail latency in half, I lowered the `vespa-sync` gRPC query deadline from 320 milliseconds to 145 milliseconds.",
        ku_s9_reply="145 milliseconds fails fast enough for the edge router to retry a mirror replica cleanly.",
        ku_question="What is the current gRPC query deadline for catalog search requests on `vespa-sync`?",
        ku_gold="145 milliseconds.",
        ku_stale_val="320",
        attr_s6_turns=(
            ("Sienna", "On the autocomplete service I set OpenTelemetry trace sampling to 15% so we catch mobile tail spikes."),
            ("Tariq", "For `vespa-sync` bulk ingestion I configured OpenTelemetry trace sampling at 2% with tail-based error capture."),
            ("Sienna", "Yeah, autocomplete QPS is lower payload per span so 15% works fine on my side."),
        ),
        attr_s8_distractor_turns=(
            ("Sienna", "My 15% OpenTelemetry sampling rate on autocomplete caught that TLS handshake regression yesterday."),
            ("Tariq", "Glad you kept that higher rate on the edge tier."),
        ),
        attr_question="What OpenTelemetry trace sampling percentage did Tariq configure for `vespa-sync` bulk ingestion?",
        attr_gold="2% (with tail-based error capture).",
        abs_s10_turns=(
            ("Tariq", "All catalog shards are migrated onto NVMe instances and the Vespa cluster upgrade is signed off."),
            ("Sienna", "Awesome, zero customer-visible blips during cutover."),
        ),
        abs_question="What replication factor did Tariq set on the Apache Cassandra keyspace?",
        abs_gold_note="Unanswerable / not mentioned: Apache Cassandra is never mentioned in the conversation.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_02",
        domain="software_engineering",
        spk_a="Mateo",
        spk_b="Freya",
        needle_context_turns=(
            ("Freya", "Did SQLite WAL checkpointing stall the UI thread on older Android devices?"),
            ("Mateo", "Only when auto-checkpoint fired mid-scroll on 10,000-row delta batches."),
            ("Mateo", "I moved `crdt-vault` checkpoints onto a dedicated background isolate and capped `journal_size_limit` at 24 megabytes."),
            ("Freya", "24 megabytes keeps local disk footprint tiny on budget phones."),
        ),
        needle_question="At how many megabytes did Mateo cap SQLite `journal_size_limit` in `crdt-vault`?",
        needle_gold="24 megabytes.",
        synth_s1_turns=(
            ("Mateo", "In the v2.1 release I stripped unused ICU locale tables and shaved 4.5 megabytes off the app binary."),
            ("Freya", "Over-the-air cellular downloads will love that."),
        ),
        synth_s4_turns=(
            ("Mateo", "For v2.3 we replaced the legacy PDF renderer and cut another 6.0 megabytes from the binary size."),
            ("Freya", "Install conversion in low-bandwidth regions already ticked up."),
        ),
        synth_s7_turns=(
            ("Mateo", "With v2.5 LTO symbols dead-stripped, we trimmed an additional 3.5 megabytes off the binary."),
            ("Freya", "Incredible how lean the release build looks now."),
        ),
        synth_question="Across releases v2.1, v2.3, and v2.5, how many total megabytes did Mateo cut from the app binary?",
        synth_gold="14 megabytes (4.5 MB in v2.1 + 6.0 MB in v2.3 + 3.5 MB in v2.5).",
        ku_s2_initial_claim="I set the background delta-sync interval in `crdt-vault` to every 15 minutes when on cellular data.",
        ku_s2_reply="Every 15 minutes seems reasonable for battery life during beta.",
        ku_echoes_6=(
            "On cellular data, `crdt-vault` runs my background delta-sync interval every 15 minutes.",
            "I have `crdt-vault` background delta-sync set to a 15-minute interval on cellular connections.",
            "Right now our cellular background delta-sync cadence in `crdt-vault` is every 15 minutes.",
            "When phones are on cellular, `crdt-vault` triggers background delta-sync every 15 minutes.",
            "I kept the cellular background delta-sync timer in `crdt-vault` at 15 minutes.",
            "`crdt-vault` still polls background delta-sync every 15 minutes over cellular data.",
        ),
        ku_echo_replies_6=(
            "Battery telemetry from TestFlight looks clean so far.",
            "Good, field technicians haven't reported stale checklists.",
            "Okay, I'll note that in the enterprise MDM docs.",
            "Makes sense alongside push-triggered wakeups.",
            "Thanks, QA verified that timer on Android 15.",
            "Cool, let's check radio wakeups after the field pilot.",
        ),
        ku_s9_update_claim="Field crews needed fresher offline handoffs, so I changed the `crdt-vault` cellular background delta-sync interval from 15 minutes down to 6 minutes.",
        ku_s9_reply="6 minutes cut conflict merge prompts by two-thirds without hurting battery.",
        ku_question="What is the current cellular background delta-sync interval in `crdt-vault`?",
        ku_gold="Every 6 minutes.",
        ku_stale_val="15",
        attr_s6_turns=(
            ("Freya", "My iOS Swift build hit a 99.84% crash-free session rate across 200k daily active users this week."),
            ("Mateo", "Awesome—my Android Kotlin build landed at a 99.61% crash-free session rate after fixing that JNI cursor leak."),
            ("Freya", "99.61% across fragmented OEM drivers is a huge milestone, and iOS staying at 99.84% gives us cushion."),
        ),
        attr_s8_distractor_turns=(
            ("Freya", "iOS held steady at 99.84% crash-free sessions again after the widget patch."),
            ("Mateo", "Nice stability heading into Q4."),
        ),
        attr_question="What crash-free session rate did Mateo report for his Android Kotlin build?",
        attr_gold="99.61%.",
        abs_s10_turns=(
            ("Mateo", "Both iOS and Android sync SDKs passed the external security pen-test with zero high findings."),
            ("Freya", "Time to tag the GA release branch."),
        ),
        abs_question="Which Linux packaging format did Mateo choose for the Flutter desktop client?",
        abs_gold_note="Unanswerable / not mentioned: neither a Flutter desktop client nor Linux desktop packaging is ever mentioned.",
        extra_closing_sessions=(
            (
                "2025-11-05 14:00",
                (
                    ("Freya", "GA rollout reached 100% of mobile devices this morning."),
                    ("Mateo", "Zero sync corruption tickets in the queue!"),
                ),
            ),
            (
                "2025-11-19 11:30",
                (
                    ("Mateo", "Archived the legacy REST polling codepath today."),
                    ("Freya", "Deleting 9,000 lines of legacy sync code feels amazing."),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_03",
        domain="software_engineering",
        spk_a="Devon",
        spk_b="Priya",
        needle_context_turns=(
            ("Priya", "How fast are cold GPU nodes joining the `triton-router` pool when traffic surges?"),
            ("Devon", "Image pull used to take four minutes until we enabled soci snapshotter lazy pulling."),
            ("Devon", "With pre-warmed NVMe AMI caches, Karpenter node startup to ready-pod is down to 47 seconds using containerd 2.0."),
            ("Priya", "47 seconds is fast enough to absorb webinar bursts without overprovisioning."),
        ),
        needle_question="How many seconds does Karpenter node startup to ready-pod take after Devon enabled pre-warmed NVMe AMI caches?",
        needle_gold="47 seconds.",
        synth_s1_turns=(
            ("Devon", "In January AWS reclaimed 9 spot GPU instances from our batch inference pool, all drained gracefully."),
            ("Priya", "Glad the termination handler caught every single one."),
        ),
        synth_s4_turns=(
            ("Devon", "April capacity crunch was tighter—we experienced 14 spot GPU preemptions across us-west-2."),
            ("Priya", "Fallback to on-demand kept queue latency steady though."),
        ),
        synth_s7_turns=(
            ("Devon", "July was calmer: only 6 spot GPU preemptions all month."),
            ("Priya", "Diversifying across three instance families clearly paid off."),
        ),
        synth_question="How many total spot GPU preemptions did Devon record across January, April, and July?",
        synth_gold="29 spot GPU preemptions (9 in January + 14 in April + 6 in July).",
        ku_s2_initial_claim="I configured the Horizontal Pod Autoscaler on `triton-router` to scale out at 72% GPU duty-cycle utilization.",
        ku_s2_reply="72% GPU utilization seems like a balanced starting threshold.",
        ku_echoes_6=(
            "Our `triton-router` HPA scales out whenever GPU duty-cycle utilization hits 72%.",
            "I have the `triton-router` autoscaler target set at 72% GPU duty-cycle utilization.",
            "Right now `triton-router` triggers HPA scale-out at my 72% GPU utilization threshold.",
            "The Horizontal Pod Autoscaler on `triton-router` is running with a 72% GPU utilization target.",
            "I kept the `triton-router` HPA threshold at 72% GPU duty-cycle utilization.",
            "`triton-router` pods still scale on the 72% GPU duty-cycle utilization rule.",
        ),
        ku_echo_replies_6=(
            "Okay, let's watch KV-cache headroom during peak hours.",
            "Sounds good for single-turn requests.",
            "Thanks, I added a horizontal line at 72% on the dashboard.",
            "Noted for the FinOps review.",
            "Got it, replica churn looked calm yesterday.",
            "Makes sense until long-context prompts ramp up.",
        ),
        ku_s9_update_claim="Long-context batches started queueing before scale-out finished, so I changed the `triton-router` HPA target from 72% down to 58% GPU duty-cycle utilization.",
        ku_s9_reply="Lowering it to 58% completely eliminated queue timeouts during the 9 AM surge.",
        ku_question="What is the current HPA GPU duty-cycle utilization target on `triton-router`?",
        ku_gold="58% GPU duty-cycle utilization.",
        ku_stale_val="72%",
        attr_s6_turns=(
            ("Priya", "On the staging observability cluster I set local Prometheus TSDB retention to 14 days before shipping blocks to Thanos."),
            ("Devon", "On the production GPU cluster I set local Prometheus TSDB retention to 5 days because high-cardinality DCGM metrics chew NVMe fast."),
            ("Priya", "True, staging has tiny cardinality so 14 days barely uses 80 GB there."),
        ),
        attr_s8_distractor_turns=(
            ("Priya", "Staging Prometheus disk stayed under 75 GB with my 14-day local retention."),
            ("Devon", "Thanos compaction is keeping up nicely on both clusters."),
        ),
        attr_question="How many days of local Prometheus TSDB retention did Devon set on the production GPU cluster?",
        attr_gold="5 days.",
        abs_s10_turns=(
            ("Devon", "Completed the quarterly Kubernetes 1.32 control-plane upgrade across staging and production."),
            ("Priya", "Zero pod disruption budget violations during the rolling drain."),
        ),
        abs_question="What certificate rotation period did Devon configure for Istio service mesh mTLS?",
        abs_gold_note="Unanswerable / not mentioned: Istio service mesh and mTLS certificate rotation are never mentioned.",
        extra_closing_sessions=(
            (
                "2025-11-14 09:40",
                (
                    ("Priya", "GPU cluster utilization report for the board deck is finalized."),
                    ("Devon", "Unit inference cost per million tokens dropped 31% since Q1."),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_04",
        domain="software_engineering",
        spk_a="Lukas",
        spk_b="Meilin",
        needle_context_turns=(
            ("Meilin", "Did Iceberg metadata manifest files blow up again on the clickstream table?"),
            ("Lukas", "Yeah, streaming micro-commits created thousands of tiny Avro manifests."),
            ("Lukas", "I scheduled `rewrite_manifests` every two hours on `flink-enrich` and set target Iceberg data file size to 512 megabytes."),
            ("Meilin", "512-megabyte data files will make Trino partition pruning fly."),
        ),
        needle_question="What target Apache Iceberg data file size did Lukas set on `flink-enrich`?",
        needle_gold="512 megabytes.",
        synth_s1_turns=(
            ("Lukas", "Backfilled the `merchant_risk` feature table in January—processed 310 million historical rows cleanly."),
            ("Meilin", "Fraud models can finally train on full twelve-month windows."),
        ),
        synth_s4_turns=(
            ("Lukas", "Finished the `cart_velocity` backfill in April, loading 190 million historical rows into Iceberg."),
            ("Meilin", "Ranking team already plugged those features into offline eval."),
        ),
        synth_s7_turns=(
            ("Lukas", "Wrapped the `device_tenure` backfill in July with 240 million historical rows validated."),
            ("Meilin", "All three core feature tables have complete point-in-time parity now."),
        ),
        synth_question="How many total million historical rows did Lukas backfill across the January, April, and July feature-table jobs?",
        synth_gold="740 million rows (310 million + 190 million + 240 million).",
        ku_s2_initial_claim="I set the Flink RocksDB incremental checkpoint interval on `flink-enrich` to 180 seconds.",
        ku_s2_reply="180 seconds keeps S3 PUT request costs very predictable.",
        ku_echoes_6=(
            "Right now `flink-enrich` runs incremental RocksDB checkpoints every 180 seconds.",
            "I have the RocksDB incremental checkpoint interval on `flink-enrich` set to 180 seconds.",
            "Our `flink-enrich` streaming job takes incremental RocksDB checkpoints on a 180-second interval.",
            "The RocksDB checkpoint cadence I configured for `flink-enrich` is 180 seconds.",
            "I kept incremental RocksDB checkpointing at 180 seconds on `flink-enrich`.",
            "`flink-enrich` is operating with a 180-second RocksDB incremental checkpoint interval.",
        ),
        ku_echo_replies_6=(
            "Checkpoint alignment duration looks under two seconds.",
            "Okay, recovery RTO stays within our five-minute target.",
            "Thanks, I'll put that in the disaster recovery doc.",
            "Good to see zero backpressure during S3 uploads.",
            "Noted for the streaming ops review.",
            "Cool, let's re-evaluate when real-time fraud scoring goes Tier-0.",
        ),
        ku_s9_update_claim="With fraud scoring moving to Tier-0 SLA, I shortened the `flink-enrich` RocksDB incremental checkpoint interval from 180 seconds to 45 seconds.",
        ku_s9_reply="45-second checkpoints brought failover replay time under 12 seconds in yesterday's game-day test.",
        ku_question="What is the current RocksDB incremental checkpoint interval on `flink-enrich`?",
        ku_gold="45 seconds.",
        ku_stale_val="180",
        attr_s6_turns=(
            ("Meilin", "In my Go feature-lookup client benchmark, p99 read latency came in at 3.8 milliseconds over gRPC."),
            ("Lukas", "In my Python asyncio feature-lookup client benchmark using uvloop, p99 read latency measured 9.4 milliseconds."),
            ("Meilin", "9.4 milliseconds from Python asyncio is still plenty fast for batch scoring, while checkout stays on my 3.8 ms Go client."),
        ),
        attr_s8_distractor_turns=(
            ("Meilin", "Checkout services using the Go client held that 3.8-millisecond p99 all week."),
            ("Lukas", "Great news for the latency budget."),
        ),
        attr_question="What p99 read latency did Lukas measure in his Python asyncio feature-lookup client benchmark?",
        attr_gold="9.4 milliseconds.",
        abs_s10_turns=(
            ("Lukas", "Decommissioned the old Hive metastore mirror now that AWS Glue catalog sync is authoritative."),
            ("Meilin", "One less stateful service to patch!"),
        ),
        abs_question="How many credits per day did Lukas budget for the Snowflake Snowpipe auto-ingest warehouse?",
        abs_gold_note="Unanswerable / not mentioned: Snowflake and Snowpipe are never mentioned in any session.",
        extra_closing_sessions=(),
    ),
    # =========================================================================
    # DOMAIN 2: HEALTH & FITNESS (5 conversations: vconv_05 .. vconv_09)
    # =========================================================================
    _ScenarioSpec(
        conv_id="vconv_05",
        domain="health_fitness",
        spk_a="Callum",
        spk_b="Zara",
        needle_context_turns=(
            ("Zara", "How did the sports physiology lab test go at the university track?"),
            ("Callum", "Super eye-opening—my aerobic threshold was higher than wrist optical HR estimated."),
            ("Callum", "The finger-stick lactate test pegged my 4.0 mmol/L anaerobic threshold pace at 4:08 per kilometer with a max heart rate of 191 bpm."),
            ("Zara", "4:08 per kilometer gives you exact tempo zones for the autumn half-marathon."),
        ),
        needle_question="What 4.0 mmol/L anaerobic threshold pace did Callum's finger-stick lactate test measure?",
        needle_gold="4:08 per kilometer.",
        synth_s1_turns=(
            ("Callum", "Wrapped January base building with 142 kilometers logged across four injury-free weeks."),
            ("Zara", "Smart to keep January aerobic and steady."),
        ),
        synth_s4_turns=(
            ("Callum", "April mileage total hit 178 kilometers after adding a mid-week medium-long run."),
            ("Zara", "Your aerobic decoupling looks way lower already."),
        ),
        synth_s7_turns=(
            ("Callum", "July peak volume block finished at 210 kilometers for the month!"),
            ("Zara", "Strongest summer base you've ever stacked."),
        ),
        synth_question="How many total kilometers did Callum log across January, April, and July combined?",
        synth_gold="530 kilometers (142 km in January + 178 km in April + 210 km in July).",
        ku_s2_initial_claim="I'm taking 200 milligrams of magnesium glycinate before bed to help with calf recovery.",
        ku_s2_reply="Evening magnesium glycinate helped my sleep quality a ton too.",
        ku_echoes_6=(
            "My nightly supplement routine includes 200 milligrams of magnesium glycinate before sleep.",
            "I still take 200 milligrams of magnesium glycinate each night for calf recovery.",
            "Before bed I'm taking 200 milligrams of magnesium glycinate.",
            "I've kept my evening magnesium glycinate dose at 200 milligrams.",
            "Right now I take a 200-milligram magnesium glycinate capsule at night.",
            "My pre-bed magnesium glycinate intake is 200 milligrams.",
        ),
        ku_echo_replies_6=(
            "Nice, any nighttime calf cramps lately?",
            "Glad your stomach tolerates glycinate well.",
            "Consistent sleep habits make all the difference.",
            "Good call during higher tempo volume.",
            "Awesome that deep sleep scores are trending up.",
            "Definitely check bloodwork at your annual physical too.",
        ),
        ku_s9_update_claim="After my sports doctor reviewed my August blood panel, I increased my nightly magnesium glycinate dose from 200 milligrams to 350 milligrams.",
        ku_s9_reply="350 milligrams makes total sense with peak sweat losses in late summer.",
        ku_question="What is Callum's current nightly magnesium glycinate dosage?",
        ku_gold="350 milligrams.",
        ku_stale_val="200",
        attr_s6_turns=(
            ("Zara", "During race-pace long runs I'm fueling at 75 grams of carbohydrates per hour using isotonic hydrogels."),
            ("Callum", "My gut couldn't handle 75 yet, so I'm fueling at 55 grams of carbohydrates per hour using maple-salt chews."),
            ("Zara", "55 grams per hour with maple-salt chews is super solid for a 90-minute half-marathon."),
        ),
        attr_s8_distractor_turns=(
            ("Zara", "Practiced my 75-gram per hour hydrogel plan again on Sunday's 24k progressive run."),
            ("Callum", "Dialing race nutrition early takes all the guesswork out."),
        ),
        attr_question="How many grams of carbohydrates per hour and what fuel type is Callum using on long runs?",
        attr_gold="55 grams of carbohydrates per hour using maple-salt chews.",
        abs_s10_turns=(
            ("Callum", "Taper week feels crisp—legs are fresh and race bib pickup is Friday afternoon."),
            ("Zara", "Go crush that sub-1:28 goal!"),
        ),
        abs_question="What simulated elevation setting did Callum use on his hypoxic altitude sleep tent?",
        abs_gold_note="Unanswerable / not mentioned: a hypoxic altitude sleep tent is never mentioned anywhere in the conversation.",
        extra_closing_sessions=(
            (
                "2025-11-09 17:10",
                (
                    ("Callum", "Crossed the half-marathon finish line in 1:27:14—negative split the back half!"),
                    ("Zara", "Huge PR!! Every bit of that patient build paid off!"),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_06",
        domain="health_fitness",
        spk_a="Henrik",
        spk_b="Ananya",
        needle_context_turns=(
            ("Ananya", "Did the bike fitter adjust your cleat fore-aft position for that left knee tightness?"),
            ("Henrik", "Yeah, moving cleats 4 mm back opened my hip angle right up."),
            ("Henrik", "He also swapped my road crankset to 165-millimeter crank arms and raised the saddle 5 millimeters."),
            ("Ananya", "165-millimeter crank arms are magic for smoother cadence over steep pitches."),
        ),
        needle_question="What crank-arm length in millimeters did the bike fitter install on Henrik's road bike?",
        needle_gold="165 millimeters.",
        synth_s1_turns=(
            ("Henrik", "January coastal ridge ride banked 1,450 meters of elevation gain despite chilly headwinds."),
            ("Ananya", "Solid winter climbing legs."),
        ),
        synth_s4_turns=(
            ("Henrik", "April canyon fondo logged 2,300 meters of elevation gain across three sustained passes."),
            ("Ananya", "That second pass has a brutal 11% ramp."),
        ),
        synth_s7_turns=(
            ("Henrik", "July alpine pass loop racked up 2,850 meters of elevation gain—biggest day on the bike yet!"),
            ("Ananya", "Nearly three thousand vertical meters in one ride is beastly."),
        ),
        synth_question="What was Henrik's combined elevation gain in meters across the January coastal ride, April canyon fondo, and July alpine pass loop?",
        synth_gold="6,600 meters (1,450 m + 2,300 m + 2,850 m).",
        ku_s2_initial_claim="My smart-trainer workouts are anchored to an FTP of 268 watts after the February ramp test.",
        ku_s2_reply="268 watts is a strong starting mark for sweet-spot intervals.",
        ku_echoes_6=(
            "I have my smart-trainer zones set to an FTP of 268 watts.",
            "Right now my cycling workouts use my 268-watt FTP setting.",
            "My Functional Threshold Power in TrainerRoad is set at 268 watts.",
            "I'm still running interval targets off my 268-watt FTP.",
            "All my indoor power zones are calibrated around an FTP of 268 watts.",
            "My current trainer FTP setting sits at 268 watts.",
        ),
        ku_echo_replies_6=(
            "Sweet-spot blocks at 90% should feel super controlled.",
            "Nice, heart-rate drift on over-unders will tell us when it moves.",
            "Good discipline not inflating it off a short 5-minute spike.",
            "Power-to-weight is climbing steadily.",
            "Awesome consistency on the Tuesday threshold sets.",
            "You're definitely due for a fresh 20-minute test soon.",
        ),
        ku_s9_update_claim="Just finished the September 20-minute outdoor climb test and updated my trainer FTP from 268 watts to 292 watts!",
        ku_s9_reply="A 24-watt jump to 292 watts is massive!",
        ku_question="What is Henrik's current Functional Threshold Power (FTP) setting in watts?",
        ku_gold="292 watts.",
        ku_stale_val="268",
        attr_s6_turns=(
            ("Ananya", "I mounted 30mm tubeless tires inflated to 58 psi on hookless carbon rims for rough chip-seal roads."),
            ("Henrik", "On my clincher wheelset I'm running 28mm tubeless tires at 66 psi since I weigh a bit more."),
            ("Ananya", "Makes total sense—58 psi on 30mm feels plush for me while 66 psi on 28mm protects your rims."),
        ),
        attr_s8_distractor_turns=(
            ("Ananya", "Zero punctures after 2,000 km on those 30mm tires at 58 psi."),
            ("Henrik", "Tubeless sealant really earned its keep this summer."),
        ),
        attr_question="What tire width and pressure in psi is Henrik running on his wheelset?",
        attr_gold="28mm tubeless tires at 66 psi.",
        abs_s10_turns=(
            ("Henrik", "Cleaned the drivetrain, waxed two chains, and switched the smart trainer into winter base mode."),
            ("Ananya", "Fresh hot-melt chain wax is so quiet indoors."),
        ),
        abs_question="What lap time did Henrik record on the indoor wooden velodrome 4-kilometer pursuit?",
        abs_gold_note="Unanswerable / not mentioned: Henrik never rode on a velodrome or did a 4-kilometer track pursuit.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_07",
        domain="health_fitness",
        spk_a="Rafael",
        spk_b="Tessa",
        needle_context_turns=(
            ("Tessa", "What scapular warmup did the sports physio give you before getting in the lap lane?"),
            ("Rafael", "External rotations with a towel roll under the elbow plus wall slides."),
            ("Rafael", "On the handheld dynamometer, external-rotation strength on my right shoulder measured 14.2 kilograms, matching 96% of my left side."),
            ("Tessa", "96% limb symmetry at 14.2 kg means that supraspinatus tendon has really calmed down."),
        ),
        needle_question="How many kilograms of external-rotation strength did the handheld dynamometer measure on Rafael's right shoulder?",
        needle_gold="14.2 kilograms.",
        synth_s1_turns=(
            ("Rafael", "Completed 6 physical therapy clinic sessions in January focused on isometric cuff loading."),
            ("Tessa", "Slow tendon remodeling pays off."),
        ),
        synth_s4_turns=(
            ("Rafael", "Logged 5 physical therapy clinic visits in April as we transitioned to overhead landmine presses."),
            ("Tessa", "Overhead pain-free range is huge progress."),
        ),
        synth_s7_turns=(
            ("Rafael", "Wrapped up my final 3 physical therapy discharge check-ins in July!"),
            ("Tessa", "Officially graduated from rehab!"),
        ),
        synth_question="Across January, April, and July, how many total physical therapy clinic visits did Rafael complete?",
        synth_gold="14 clinic visits (6 in January + 5 in April + 3 in July).",
        ku_s2_initial_claim="My physio capped my weekly freestyle swim volume at 1,800 yards while rebuilding shoulder tolerance.",
        ku_s2_reply="1,800 yards a week with fins keeps stroke mechanics crisp without fatigue.",
        ku_echoes_6=(
            "Right now my weekly freestyle swim cap is set at 1,800 yards.",
            "I'm keeping my total weekly pool yardage at the 1,800-yard limit.",
            "My shoulder return-to-swim plan caps me at 1,800 yards per week.",
            "I still swim up to 1,800 yards a week under my rehab cap.",
            "Weekly pool volume for me remains capped at 1,800 yards.",
            "I have my weekly freestyle limit held at 1,800 yards.",
        ),
        ku_echo_replies_6=(
            "Zero next-morning soreness is the key test.",
            "Smart restraint—swimming fitness comes back quick once healthy.",
            "High-elbow catch drills look super smooth.",
            "Glad the pull buoy didn't irritate anything.",
            "Consistency beats rushing yardage every time.",
            "Almost ready for masters lane workouts again!",
        ),
        ku_s9_update_claim="With eight pain-free weeks behind me, my physio raised my weekly freestyle swim volume cap from 1,800 yards to 4,200 yards!",
        ku_s9_reply="4,200 yards a week puts full three-day masters workouts right back on the menu!",
        ku_question="What is Rafael's current weekly freestyle swim volume cap in yards?",
        ku_gold="4,200 yards.",
        ku_stale_val="1,800",
        attr_s6_turns=(
            ("Tessa", "My favorite technique set in the 25-meter pool is 6x100 single-arm backstroke with Tempo Trainer beeps."),
            ("Rafael", "For me, my go-to technique set is 8x75 fingertip-drag freestyle using a center-mount snorkel."),
            ("Tessa", "Center-mount snorkel takes neck rotation completely out of the equation while grooving catch."),
        ),
        attr_s8_distractor_turns=(
            ("Tessa", "Did that 6x100 single-arm backstroke Tempo Trainer set again this morning."),
            ("Rafael", "Nothing exposes a dead spot in the pull faster."),
        ),
        attr_question="What is Rafael's go-to swim technique set and equipment?",
        attr_gold="8x75 fingertip-drag freestyle using a center-mount snorkel.",
        abs_s10_turns=(
            ("Rafael", "Swam a continuous 1,500-yard time trial Saturday morning with zero shoulder tightness afterward."),
            ("Tessa", "Full comeback achieved!"),
        ),
        abs_question="On what date did Rafael receive a subacromial cortisone injection?",
        abs_gold_note="Unanswerable / not mentioned: Rafael never received or mentioned a cortisone injection.",
        extra_closing_sessions=(
            (
                "2025-11-18 08:20",
                (
                    ("Tessa", "Signed up for the spring 2-mile lake swim yet?"),
                    ("Rafael", "Registered yesterday—early bird wave!"),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_08",
        domain="health_fitness",
        spk_a="Darius",
        spk_b="Linnea",
        needle_context_turns=(
            ("Linnea", "Did switching from mixed grip to hook grip tear up your thumbs on heavy pulls?"),
            ("Darius", "Only for the first ten days until I learned to tape the interphalangeal joint properly."),
            ("Darius", "I've been using Johnson & Johnson Coach cloth athletic tape torn to 19-millimeter strips and zero chalk slip."),
            ("Linnea", "19-millimeter strips bend cleanly around the thumb knuckle without bunching."),
        ),
        needle_question="What width of athletic tape strips in millimeters does Darius use for hook-grip thumb taping?",
        needle_gold="19-millimeter strips.",
        synth_s1_turns=(
            ("Darius", "End of Mesocycle 1: added 7.5 kilograms to my competition pause bench press."),
            ("Linnea", "Paused reps off the chest looked rockets."),
        ),
        synth_s4_turns=(
            ("Darius", "Mesocycle 2 test day added 12.5 kilograms to my high-bar back squat!"),
            ("Linnea", "Depth was buried too—three white lights anywhere."),
        ),
        synth_s7_turns=(
            ("Darius", "Mesocycle 3 peaked cleanly and added 15.0 kilograms to my conventional deadlift."),
            ("Linnea", "Massive progress across all three lifts this year."),
        ),
        synth_question="Combining Darius's bench press gain in Mesocycle 1, squat gain in Mesocycle 2, and deadlift gain in Mesocycle 3, how many total kilograms did he add to his three-lift total?",
        synth_gold="35 kilograms (7.5 kg + 12.5 kg + 15.0 kg).",
        ku_s2_initial_claim="I'm hitting 35 grams of whey isolate protein at breakfast right after morning mobility.",
        ku_s2_reply="Getting leucine in early sets up recovery for the whole day.",
        ku_echoes_6=(
            "Every morning I start with 35 grams of whey isolate protein at breakfast.",
            "My breakfast protein target is 35 grams from whey isolate.",
            "Right now I take 35 grams of whey isolate with breakfast.",
            "I've kept my morning breakfast whey isolate intake at 35 grams.",
            "First meal of the day for me still has 35 grams of whey isolate protein.",
            "I get 35 grams of whey isolate protein each morning at breakfast.",
        ),
        ku_echo_replies_6=(
            "Easy to digest before morning meetings.",
            "Unflavored or vanilla bean?",
            "Consistent meal timing shows in your recovery HRV.",
            "Makes hitting daily macros effortless.",
            "Nice, soreness after squat days seems way lower.",
            "Let's see if you adjust during the autumn lean bulk.",
        ),
        ku_s9_update_claim="For my autumn hypertrophy block I bumped my breakfast protein intake from 35 grams up to 50 grams using whey isolate plus Greek yogurt.",
        ku_s9_reply="50 grams at breakfast gives awesome satiety through lunchtime too.",
        ku_question="How many grams of protein is Darius currently getting at breakfast?",
        ku_gold="50 grams.",
        ku_stale_val="35",
        attr_s6_turns=(
            ("Linnea", "My Oura ring 30-day baseline shows a 48 bpm resting heart rate with a strict 22:15 lights-out target."),
            ("Darius", "On my Garmin watch my 30-day average resting heart rate is 54 bpm and I keep a 22:45 lights-out target."),
            ("Linnea", "54 bpm for a heavyweight powerlifter is great cardiovascular conditioning."),
        ),
        attr_s8_distractor_turns=(
            ("Linnea", "My resting heart rate dipped to 47 bpm this week sticking to that 22:15 lights-out schedule."),
            ("Darius", "Sleep regularity beats every recovery gadget on the market."),
        ),
        attr_question="What 30-day average resting heart rate and lights-out target did Darius report?",
        attr_gold="54 bpm resting heart rate and a 22:45 lights-out target.",
        abs_s10_turns=(
            ("Darius", "Weighed in right at 92.4 kg for Saturday's local USAPL mock meet."),
            ("Linnea", "Comfortably under the 93 kg class cap with zero water cut!"),
        ),
        abs_question="How many grams per day did Darius take during his 7-day creatine monohydrate loading phase?",
        abs_gold_note="Unanswerable / not mentioned: creatine monohydrate and a loading phase are never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_09",
        domain="health_fitness",
        spk_a="Gideon",
        spk_b="Soraya",
        needle_context_turns=(
            ("Soraya", "Did your waist-mounted trail light bounce at all on rocky technical descents at night?"),
            ("Gideon", "Zero bounce—the wide beam casts shadows across tree roots way better than a forehead lamp alone."),
            ("Gideon", "My primary waist lamp runs at 600 sustained lumens for 6 hours on a single 21700 lithium cell."),
            ("Soraya", "600 sustained lumens on one 21700 cell means only carrying a single spare battery in the drop bag."),
        ),
        needle_question="How many sustained lumens does Gideon's primary waist lamp output, and on what battery cell?",
        needle_gold="600 sustained lumens on a single 21700 lithium cell.",
        synth_s1_turns=(
            ("Gideon", "January pine-barrens scouting loop covered 26 kilometers with damp sandy footing."),
            ("Soraya", "Great ankle stabilizer work early in the season."),
        ),
        synth_s4_turns=(
            ("Gideon", "April granite ridge recon hike-run totaled 34 kilometers back-to-back with storm gear."),
            ("Soraya", "Testing rain shells in real sleet is priceless."),
        ),
        synth_s7_turns=(
            ("Gideon", "July high-basin dress-rehearsal run logged 42 kilometers at race effort."),
            ("Soraya", "Full marathon distance on technical singletrack—you're ready for the 50-miler."),
        ),
        synth_question="What was the combined distance in kilometers of Gideon's January, April, and July scouting runs?",
        synth_gold="102 kilometers (26 km + 34 km + 42 km).",
        ku_s2_initial_claim="I'm mixing my trail flasks to 650 milligrams of sodium per liter of water.",
        ku_s2_reply="650 milligrams per liter works nicely in cool spring weather.",
        ku_echoes_6=(
            "My trail hydration flasks are mixed at 650 milligrams of sodium per liter.",
            "Right now I run 650 milligrams of sodium per liter in my soft flasks.",
            "I've kept my electrolyte mix at 650 milligrams of sodium per liter of water.",
            "Each liter in my running vest still carries 650 milligrams of sodium.",
            "For long trail outings I mix 650 milligrams of sodium per liter.",
            "My flask sodium concentration is set at 650 milligrams per liter.",
        ),
        ku_echo_replies_6=(
            "No puffy fingers at the end of four-hour efforts?",
            "Good baseline before the humid midsummer heat arrives.",
            "Plain water in one flask and mix in the other is handy too.",
            "Sweat-patch test next month will give exact numbers.",
            "Nice that taste fatigue hasn't been an issue.",
            "Let's see what the microfluidic sweat test says.",
        ),
        ku_s9_update_claim="My sweat-sodium patch test came back salty, so I raised my trail flask mix from 650 milligrams up to 950 milligrams of sodium per liter.",
        ku_s9_reply="950 milligrams per liter completely stopped those late-climb quad twinges!",
        ku_question="What sodium concentration in milligrams per liter does Gideon currently mix in his trail flasks?",
        ku_gold="950 milligrams of sodium per liter.",
        ku_stale_val="650",
        attr_s6_turns=(
            ("Soraya", "I'm racing in 4mm-drop Vibram Megagrip shoes paired with a 12-liter Solomon hydration vest."),
            ("Gideon", "I settled on 6mm-drop rocker trail shoes paired with an 8-liter Raide waist-and-vest pack."),
            ("Soraya", "6mm drop saves the Achilles on steep late-race climbs."),
        ),
        attr_s8_distractor_turns=(
            ("Soraya", "My 4mm-drop Vibram shoes gripped wet slate slabs effortlessly on Saturday."),
            ("Gideon", "Confidence on wet rock saves so much braking energy."),
        ),
        attr_question="What shoe heel-to-toe drop and pack capacity did Gideon settle on?",
        attr_gold="6mm-drop rocker trail shoes and an 8-liter Raide pack.",
        abs_s10_turns=(
            ("Gideon", "Drop bags are labeled for Mile 22 and Mile 38—flying out Thursday morning!"),
            ("Soraya", "Have an unforgettable day out on the ridge!"),
        ),
        abs_question="Which brand of folding Z-fold carbon trekking poles did Gideon pack in his Mile 22 drop bag?",
        abs_gold_note="Unanswerable / not mentioned: trekking poles are never mentioned in any session.",
        extra_closing_sessions=(),
    ),
    # =========================================================================
    # DOMAIN 3: TRAVEL (5 conversations: vconv_10 .. vconv_14)
    # =========================================================================
    _ScenarioSpec(
        conv_id="vconv_10",
        domain="travel",
        spk_a="Kenji",
        spk_b="Astrid",
        needle_context_turns=(
            ("Astrid", "Should we haul our big snow duffels onto the local train between New Chitose and Otaru?"),
            ("Kenji", "Way easier to ship them ahead from the airport Yamato Kuroneko counter so our hands are free."),
            ("Kenji", "Yamato quoted 2,640 yen per large ski bag for overnight delivery straight to our Sapporo hotel bell desk."),
            ("Astrid", "2,640 yen per bag is a total bargain to skip icy station stairs."),
        ),
        needle_question="How many yen per large ski bag did Yamato quote Kenji for overnight delivery to Sapporo?",
        needle_gold="2,640 yen per bag.",
        synth_s1_turns=(
            ("Kenji", "Locked in 3 nights at the canal-side ryokan in Otaru for the first leg."),
            ("Astrid", "Snow lanterns along the canal in February look magical."),
        ),
        synth_s4_turns=(
            ("Kenji", "Added 4 nights in Asahikawa as our base for Daisetsuzan powder and ramen alleys."),
            ("Astrid", "Asahikawa ramen after sub-zero mornings is unbeatable."),
        ),
        synth_s7_turns=(
            ("Kenji", "Booked 2 nights in Hakodate at the end of the route for the morning seafood market and Mount Hakodate view."),
            ("Astrid", "Kaisendon breakfast right by the harbor!"),
        ),
        synth_question="Counting Otaru, Asahikawa, and Hakodate together, how many total nights did Kenji book across those three towns?",
        synth_gold="9 nights (3 in Otaru + 4 in Asahikawa + 2 in Hakodate).",
        ku_s2_initial_claim="For our Sapporo stopover I reserved a twin room at Hotel Resol Trinity Odori.",
        ku_s2_reply="Right across from Odori Park—super convenient for the snow sculptures.",
        ku_echoes_6=(
            "Our Sapporo reservation is a twin room at Hotel Resol Trinity Odori.",
            "I have us booked at Hotel Resol Trinity Odori for the Sapporo leg.",
            "Right now we're staying at Hotel Resol Trinity Odori while in Sapporo.",
            "For Sapporo I'm holding our twin booking at Hotel Resol Trinity Odori.",
            "Our Sapporo hotel remains Hotel Resol Trinity Odori by the park.",
            "We have a twin room reserved at Hotel Resol Trinity Odori in Sapporo.",
        ),
        ku_echo_replies_6=(
            "Great rooftop bath there too.",
            "Easy walk to the underground walkway if a blizzard hits.",
            "Thanks, I pinned the address in Google Maps.",
            "Perfect for catching the airport bus.",
            "Free cancellation window goes till autumn, right?",
            "Let me know if direct underground station access opens up elsewhere.",
        ),
        ku_s9_update_claim="A corner room with direct underground walkway access opened up, so I switched our Sapporo reservation from Hotel Resol Trinity Odori to Solaria Nishitetsu Hotel Sapporo.",
        ku_s9_reply="Solaria Nishitetsu right by JR Sapporo Station makes early morning train departures way smoother!",
        ku_question="Which hotel is currently booked for Kenji and Astrid's Sapporo stay?",
        ku_gold="Solaria Nishitetsu Hotel Sapporo.",
        ku_stale_val="Resol Trinity",
        attr_s6_turns=(
            ("Astrid", "For Furano day two I booked a guided backcountry snowshoe tour through the birch forest."),
            ("Kenji", "Nice—for that same morning I reserved demo powder skis (108mm underfoot) at Rhythm Furano."),
            ("Astrid", "Meeting back at the base cafe at 2 PM for katsu curry sounds ideal."),
        ),
        attr_s8_distractor_turns=(
            ("Astrid", "The Furano guide confirmed snowshoe crampons and poles are included in my tour."),
            ("Kenji", "Awesome, one less thing to rent."),
        ),
        attr_question="What activity and gear did Kenji reserve for himself on Furano day two?",
        attr_gold="Demo powder skis (108mm underfoot) at Rhythm Furano.",
        abs_s10_turns=(
            ("Kenji", "JR Hokkaido 7-day rail passes and Visit Japan Web QR codes are all saved offline."),
            ("Astrid", "Bring on the Hokkaido winter!"),
        ),
        abs_question="Which wildlife photography blind did Kenji reserve at the Kushiro Marsh red-crowned crane sanctuary?",
        abs_gold_note="Unanswerable / not mentioned: Kushiro Marsh and the red-crowned crane sanctuary are never mentioned.",
        extra_closing_sessions=(
            (
                "2025-11-16 18:00",
                (
                    ("Astrid", "Picked up Merino base layers and ice cleats for city sidewalks."),
                    ("Kenji", "Smart move—Sapporo crosswalks get super slick at dusk."),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_11",
        domain="travel",
        spk_a="Nikolai",
        spk_b="Paloma",
        needle_context_turns=(
            ("Paloma", "Do we need full climbing harnesses for the Via Ferrata Averau section?"),
            ("Nikolai", "Yes, lightweight alpine loop harness, helmet, and an energy-absorbing Y-lanyard meeting EN 958."),
            ("Nikolai", "I picked up the Edelrid Cable Comfort VI via ferrata lanyard set—it weighs 485 grams with elastic arms."),
            ("Paloma", "485 grams barely registers in the lid pocket."),
        ),
        needle_question="Which exact via ferrata lanyard model did Nikolai buy and how many grams does it weigh?",
        needle_gold="The Edelrid Cable Comfort VI, weighing 485 grams.",
        synth_s1_turns=(
            ("Nikolai", "Wired the 90-euro half-board deposit to Rifugio Fanes for night one on the Alta Via."),
            ("Paloma", "Rifugio Fanes looks like a postcard amphitheater."),
        ),
        synth_s4_turns=(
            ("Nikolai", "Paid the 110-euro half-board deposit for Rifugio Lagazuoi right on the summit crest."),
            ("Paloma", "Sunset over Marmolada from that terrace is legendary."),
        ),
        synth_s7_turns=(
            ("Nikolai", "Sent the final 85-euro deposit to Rifugio Nuvolau for our third mountain hut night."),
            ("Paloma", "Three iconic rifugios locked in!"),
        ),
        synth_question="How many total euros in hut deposits did Nikolai pay across Rifugio Fanes, Rifugio Lagazuoi, and Rifugio Nuvolau?",
        synth_gold="285 euros (90 EUR + 110 EUR + 85 EUR).",
        ku_s2_initial_claim="I booked our Venice-to-Cortina ATVO express coach for the 08:15 departure from Mestre station.",
        ku_s2_reply="08:15 gets us into Cortina before noon.",
        ku_echoes_6=(
            "Our ATVO coach from Venice Mestre to Cortina is booked for the 08:15 departure.",
            "I have tickets on the 08:15 ATVO express bus out of Mestre to Cortina.",
            "Right now we're scheduled on the 08:15 Venice Mestre coach up to Cortina.",
            "We're taking the 08:15 ATVO departure from Mestre station to Cortina.",
            "I kept our Mestre-to-Cortina express coach seat reservation at 08:15.",
            "Our bus ticket up to Cortina leaves Venice Mestre at 08:15.",
        ),
        ku_echo_replies_6=(
            "Leaves time to buy stove fuel canisters in town.",
            "Great, platform 1 bus bay is right outside Mestre exit.",
            "Thanks, I added the booking reference to our shared folder.",
            "Hope the mountain pass views are clear that morning.",
            "Plenty of luggage hold space underneath ATVO coaches.",
            "Wait—did our inbound train schedule shift by half an hour?",
        ),
        ku_s9_update_claim="Because Trenitalia shifted our regional train arrival, I changed our Mestre-to-Cortina ATVO coach ticket from 08:15 to the 10:40 departure.",
        ku_s9_reply="10:40 gives us a stress-free 55-minute transfer with cappuccino time!",
        ku_question="What departure time is Nikolai and Paloma's current Mestre-to-Cortina ATVO coach reservation?",
        ku_gold="10:40.",
        ku_stale_val="08:15",
        attr_s6_turns=(
            ("Paloma", "Weighed my fully loaded Osprey Tempest 34 pack this morning—6.8 kilograms dry before water."),
            ("Nikolai", "My Deuter Futura Pro 36 weighed in at 7.9 kilograms dry since I'm carrying the group first-aid kit and Tabacco paper maps."),
            ("Paloma", "7.9 kilograms including full emergency kit and paper maps is still super trim for hut-to-hut."),
        ),
        attr_s8_distractor_turns=(
            ("Paloma", "Trimmed an extra tee so my Tempest 34 stayed right at 6.8 kilograms dry."),
            ("Nikolai", "Knees will thank us on those 1,100-meter scree descents."),
        ),
        attr_question="Which backpack model and dry weight did Nikolai report?",
        attr_gold="Deuter Futura Pro 36 at 7.9 kilograms dry.",
        abs_s10_turns=(
            ("Nikolai", "Alpenverein rescue insurance cards arrived in the mail and boots are broken in."),
            ("Paloma", "Dolomites limestone awaits!"),
        ),
        abs_question="What timed entry slot did Nikolai reserve for the South Tyrol Museum of Archaeology (Ötzi) in Bolzano?",
        abs_gold_note="Unanswerable / not mentioned: Bolzano and the South Tyrol Museum of Archaeology (Ötzi) are never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_12",
        domain="travel",
        spk_a="Idris",
        spk_b="Camille",
        needle_context_turns=(
            ("Camille", "Which marine biologist crew did you pick for sperm-whale spotting off São Miguel?"),
            ("Idris", "Went with Terra Azul out of Vila Franca do Campo marina—they use land-based vigia cliff spotters instead of sonar."),
            ("Idris", "Our semi-rigid zodiac check-in is at Pier B, Slip 14, with hydrophone listening included."),
            ("Camille", "Pier B, Slip 14 in Vila Franca—right next to the queijadas pastry shop!"),
        ),
        needle_question="Which marina pier and slip number is Idris's Terra Azul whale-watching check-in at?",
        needle_gold="Pier B, Slip 14 (at Vila Franca do Campo marina).",
        synth_s1_turns=(
            ("Idris", "Booked our SATA Azores Airlines hop from São Miguel to Pico for 118 euros total."),
            ("Camille", "Views of Mount Pico poking through the clouds on approach look unreal."),
        ),
        synth_s4_turns=(
            ("Idris", "Reserved the Atlânticoline ferry crossing from Pico over to São Jorge for 36 euros."),
            ("Camille", "Short 50-minute channel crossing for fajã hikes!"),
        ),
        synth_s7_turns=(
            ("Idris", "Locked in our return SATA flight from São Jorge back to Ponta Delgada for 134 euros."),
            ("Camille", "All three inter-island hops sorted cleanly."),
        ),
        synth_question="What was the combined cost in euros of the three inter-island transport bookings Idris made (São Miguel-Pico flight, Pico-São Jorge ferry, and São Jorge-Ponta Delgada flight)?",
        synth_gold="288 euros (118 EUR + 36 EUR + 134 EUR).",
        ku_s2_initial_claim="I reserved a compact hatchback through Ilha Verde Rent-a-Car at Ponta Delgada airport.",
        ku_s2_reply="Compact dimensions are essential for narrow hydrangeas-lined village lanes.",
        ku_echoes_6=(
            "Our Ponta Delgada rental car is a compact hatchback booked with Ilha Verde Rent-a-Car.",
            "I have our São Miguel car reservation with Ilha Verde Rent-a-Car at the airport.",
            "Right now we're renting from Ilha Verde Rent-a-Car when we land in Ponta Delgada.",
            "For São Miguel drives I'm holding our compact booking at Ilha Verde Rent-a-Car.",
            "Our airport rental car remains booked through Ilha Verde Rent-a-Car.",
            "We have a hatchback reserved at Ilha Verde Rent-a-Car in Ponta Delgada.",
        ),
        ku_echo_replies_6=(
            "Full zero-deductible CDW is worth it on volcanic stone walls.",
            "Great, Sete Cidades caldera viewpoint early beats the tour buses.",
            "Thanks, I downloaded offline maps of São Miguel.",
            "Automatic transmission or manual?",
            "Good call—parking in Furnas is tight.",
            "Did you see Wayzor includes zero-deposit gravel-road coverage?",
        ),
        ku_s9_update_claim="Wayzor offered zero-deductible coverage with no security hold, so I switched our Ponta Delgada rental car reservation from Ilha Verde Rent-a-Car to Wayzor Rent-a-Car.",
        ku_s9_reply="Wayzor having shuttleless terminal pickup and zero deposit hold is way cleaner!",
        ku_question="Which rental car company is Idris and Camille's current Ponta Delgada reservation with?",
        ku_gold="Wayzor Rent-a-Car.",
        ku_stale_val="Ilha Verde",
        attr_s6_turns=(
            ("Camille", "On São Jorge I signed up for the Ribeira doarem waterfall canyoning descent with 5mm wetsuits."),
            ("Idris", "Nice—while you canyon, I booked a two-tank nitrox scuba dive at Princess Alice Bank setup out of Pico."),
            ("Camille", "Mobula rays in blue water for you and volcanic waterfalls for me!"),
        ),
        attr_s8_distractor_turns=(
            ("Camille", "Confirmed canyoning boots are provided for my São Jorge waterfall descent."),
            ("Idris", "My dive computer battery passed its pressure check too."),
        ),
        attr_question="What excursion did Idris book for himself while Camille goes canyoning?",
        attr_gold="A two-tank nitrox scuba dive (out of Pico / Princess Alice Bank setup).",
        abs_s10_turns=(
            ("Idris", "Packed lightweight rain shells and natural hot-spring dark swimwear for Terra Nostra iron pools."),
            ("Camille", "Good tip—orange iron water stains bright suits instantly!"),
        ),
        abs_question="What time is Idris's guided tour inside the Algar do Carvão lava tube on Terceira Island?",
        abs_gold_note="Unanswerable / not mentioned: Terceira Island and Algar do Carvão lava tube are never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_13",
        domain="travel",
        spk_a="Bastian",
        spk_b="Yael",
        needle_context_turns=(
            ("Yael", "Patagonian gusts hit 90 km/h—did you upgrade your rain shell before we leave?"),
            ("Bastian", "Definitely, my old 2.5-layer jacket wetted out in Scotland last year."),
            ("Bastian", "I bought a 3-layer Gore-Tex Pro hardshell rated to a 28,000-millimeter hydrostatic head with pit zips."),
            ("Yael", "28,000 millimeters hydrostatic head will shrug off sideways sleet in French Valley."),
        ),
        needle_question="What hydrostatic-head waterproof rating in millimeters does Bastian's new 3-layer hardshell have?",
        needle_gold="28,000 millimeters.",
        synth_s1_turns=(
            ("Bastian", "Mapped our El Chaltén warmup day: Laguna Torre out-and-back is 18 kilometers round-trip."),
            ("Yael", "Cerro Torre spires over the glacier lake on day one!"),
        ),
        synth_s4_turns=(
            ("Bastian", "Day two weather window target: Laguna de los Tres up to Fitz Roy viewpoint clocks in at 22 kilometers."),
            ("Yael", "That final steep kilometer over moraine boulders is iconic."),
        ),
        synth_s7_turns=(
            ("Bastian", "Added Loma del Pliegue Tumbado panoramic ridge at 20 kilometers for our third El Chaltén day hike."),
            ("Yael", "Best 360-degree vantage point over both valleys."),
        ),
        synth_question="How many total kilometers are the three planned El Chaltén day hikes (Laguna Torre, Laguna de los Tres, and Loma del Pliegue Tumbado) combined?",
        synth_gold="60 kilometers (18 km + 22 km + 20 km).",
        ku_s2_initial_claim="I booked seats on the 07:00 Bus-Sur coach from Puerto Natales into Torres del Paine Laguna Amarga.",
        ku_s2_reply="07:00 connects cleanly with the Pudeto catamaran.",
        ku_echoes_6=(
            "Our Bus-Sur tickets from Puerto Natales to Laguna Amarga are for the 07:00 departure.",
            "I have us reserved on the 07:00 Bus-Sur coach into Torres del Paine.",
            "Right now we're riding the 07:00 Bus-Sur bus out of Puerto Natales.",
            "For park entry day we hold seats on the 07:00 Bus-Sur departure.",
            "Our Puerto Natales coach into Laguna Amarga remains the 07:00 Bus-Sur service.",
            "We're booked on Bus-Sur leaving Puerto Natales terminal at 07:00.",
        ),
        ku_echo_replies_6=(
            "Hotel packs an early breakfast box from 06:00.",
            "Terminal Rodoviario is only ten minutes from our hostel.",
            "Thanks, printed two paper copies of the CONAF park QR codes.",
            "Window seats on the left side get the first Paine massif view.",
            "Remember Chilean pesos cash for the Pudeto catamaran.",
            "Wait—CONAF opened an afternoon check-in slot at Refugio Paine Grande!",
        ),
        ku_s9_update_claim="Since Refugio Paine Grande confirmed late afternoon check-in, I moved our Puerto Natales Bus-Sur coach reservation from 07:00 to the 12:30 departure.",
        ku_s9_reply="12:30 lets us pick up fresh empanadas in town and catch the golden-hour catamaran!",
        ku_question="What is the current departure time of Bastian and Yael's Bus-Sur coach from Puerto Natales?",
        ku_gold="12:30.",
        ku_stale_val="07:00",
        attr_s6_turns=(
            ("Yael", "I'm bringing a Sony 16-35mm f/4 wide-angle zoom so granite towers fit in frame from shore."),
            ("Bastian", "Good pairing—I'm packing a 70-200mm f/4 telephoto lens to isolate alpenglow on the summits and Andean condors."),
            ("Yael", "Zero focal-length overlap between my 16-35mm and your 70-200mm!"),
        ),
        attr_s8_distractor_turns=(
            ("Yael", "Bought a circular polarizer filter for my 16-35mm wide-angle lens to cut glacial lake glare."),
            ("Bastian", "Turquoise meltwater colors are going to pop."),
        ),
        attr_question="Which camera lens focal length and aperture is Bastian packing?",
        attr_gold="70-200mm f/4 telephoto lens.",
        abs_s10_turns=(
            ("Bastian", "Border-crossing reciprocity docs for Argentina and Chile are tucked into our passport sleeves."),
            ("Yael", "Southern hemisphere summer, here we come!"),
        ),
        abs_question="Which expedition ship cabin category did Bastian book out of Ushuaia for the Antarctica Drake Passage cruise?",
        abs_gold_note="Unanswerable / not mentioned: Ushuaia and an Antarctica cruise are never mentioned.",
        extra_closing_sessions=(
            (
                "2025-11-11 19:15",
                (
                    ("Yael", "Tested waterproof dry-sacks in the shower—bone dry."),
                    ("Bastian", "Down sleeping bags stay lofty no matter what."),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_14",
        domain="travel",
        spk_a="Stellan",
        spk_b="Noemi",
        needle_context_turns=(
            ("Noemi", "Did you snag early morning entry to the Mosque-Cathedral in Córdoba before tour groups arrive?"),
            ("Stellan", "Even better—during the 08:30 bell-tower and patio window it's super peaceful."),
            ("Stellan", "Our Mezquita Bell Tower climb ticket is for the 09:00 timed group of 20 visitors, 193 steps to the top."),
            ("Noemi", "09:00 right after watching morning light through the horseshoe arches is perfection."),
        ),
        needle_question="What timed slot and step count did Stellan note for the Córdoba Mezquita Bell Tower climb?",
        needle_gold="09:00 timed slot (193 steps).",
        synth_s1_turns=(
            ("Stellan", "Grabbed Renfe promo fares for Madrid to Córdoba AVE high-speed rail at 54 euros for the two of us."),
            ("Noemi", "Under two hours center-to-center beats flying any day."),
        ),
        synth_s4_turns=(
            ("Stellan", "Booked our Córdoba to Seville Avant rail leg for 38 euros total."),
            ("Noemi", "45 minutes door to door into Santa Justa!"),
        ),
        synth_s7_turns=(
            ("Stellan", "Added the Seville to Granada Avant train tickets at 68 euros for both seats."),
            ("Noemi", "Smooth rail triangle all across Andalusia."),
        ),
        synth_question="What was the combined rail ticket cost in euros across all three train legs Stellan booked (Madrid-Córdoba, Córdoba-Seville, and Seville-Granada)?",
        synth_gold="160 euros (54 EUR + 38 EUR + 68 EUR).",
        ku_s2_initial_claim="I secured our Alhambra Nasrid Palaces timed entry in Granada for the 11:30 morning slot.",
        ku_s2_reply="Glad we booked three months ahead—tickets vanish fast.",
        ku_echoes_6=(
            "Our Alhambra Nasrid Palaces entry ticket is set for the 11:30 slot.",
            "I have our timed Nasrid Palaces pass in Granada booked for 11:30.",
            "Right now our Alhambra Nasrid Palaces admission time is 11:30.",
            "We're entering the Nasrid Palaces at the Alhambra on our 11:30 ticket.",
            "I'm holding our 11:30 timed entry slot for the Nasrid Palaces.",
            "Our Granada Alhambra Nasrid Palaces reservation is for 11:30.",
        ),
        ku_echo_replies_6=(
            "Remember original physical passports—they scan IDs at the palace gate!",
            "We can stroll the Generalife gardens before or after.",
            "Thanks, saved the Patronato PDFs to both phones.",
            "Albaicín mirador at sunset afterward will be gorgeous.",
            "Comfortable walking shoes for those cobblestone hills.",
            "Oh look—the Patronato just released evening night-visit tickets!",
        ),
        ku_s9_update_claim="Patronato released nocturnal palace passes and let us exchange, so I switched our Nasrid Palaces entry from 11:30 to the 20:00 evening illuminated slot!",
        ku_s9_reply="Seeing the Court of the Lions lit at 20:00 without midday crowds is a dream!",
        ku_question="What timed entry slot do Stellan and Noemi currently hold for the Nasrid Palaces at the Alhambra?",
        ku_gold="20:00 (evening illuminated slot).",
        ku_stale_val="11:30",
        attr_s6_turns=(
            ("Noemi", "In Seville's Triana quarter I booked an intimate 50-seat traditional flamenco tablao performance."),
            ("Stellan", "And for our free afternoon in Seville I booked a hands-on salmorejo and Iberian tapas market cooking workshop."),
            ("Noemi", "Market cooking in the afternoon and Triana flamenco after dinner—peak Seville day!"),
        ),
        attr_s8_distractor_turns=(
            ("Noemi", "The Triana flamenco tablao confirmed acoustic guitar and cante jondo with zero microphones."),
            ("Stellan", "Goosebumps guaranteed in a 50-seat courtyard."),
        ),
        attr_question="What cultural activity did Stellan book in Seville?",
        attr_gold="A hands-on salmorejo and Iberian tapas market cooking workshop.",
        abs_s10_turns=(
            ("Stellan", "Added a final afternoon at the Prado in Madrid before our flight home."),
            ("Noemi", "Velázquez and Goya to close out the trip!"),
        ),
        abs_question="Which ferry operator did Stellan book from Tarifa/Cádiz across the Strait of Gibraltar to Tangier?",
        abs_gold_note="Unanswerable / not mentioned: neither Cádiz/Tarifa nor a ferry to Tangier is ever mentioned.",
        extra_closing_sessions=(),
    ),
    # =========================================================================
    # DOMAIN 4: PERSONAL FINANCE (5 conversations: vconv_15 .. vconv_19)
    # =========================================================================
    _ScenarioSpec(
        conv_id="vconv_15",
        domain="personal_finance",
        spk_a="Declan",
        spk_b="Marisol",
        needle_context_turns=(
            ("Marisol", "Did the credit union waive the origination fee on our 15-year mortgage refinance?"),
            ("Declan", "Yes, zero origination points and they locked us at 5.125% fixed."),
            ("Declan", "Our only out-of-pocket upfront charge before closing was the independent home appraisal fee of $525."),
            ("Marisol", "$525 total upfront with zero points pays back in under five months."),
        ),
        needle_question="How much was the upfront independent home appraisal fee on Declan and Marisol's mortgage refinance?",
        needle_gold="$525.",
        synth_s1_turns=(
            ("Declan", "Placed Rung 1 of our emergency Treasury ladder in January: $12,000 in 13-week T-bills at TreasuryDirect."),
            ("Marisol", "State-tax-exempt yield beats our old brick-and-mortar savings account easily."),
        ),
        synth_s4_turns=(
            ("Declan", "Added Rung 2 in April's auction with $15,000 in 26-week T-bills."),
            ("Marisol", "Staggered maturities are coming together nicely."),
        ),
        synth_s7_turns=(
            ("Declan", "Completed Rung 3 in July by purchasing $18,000 in 52-week T-bills."),
            ("Marisol", "Full three-rung liquidity reserve is officially live."),
        ),
        synth_question="What is the combined principal across all three Treasury bill rungs Declan purchased in January, April, and July?",
        synth_gold="$45,000 ($12,000 + $15,000 + $18,000).",
        ku_s2_initial_claim="I set up an automatic monthly extra principal payment of $350 on our mortgage.",
        ku_s2_reply="$350 extra every month shaves years off the amortization schedule.",
        ku_echoes_6=(
            "Our mortgage auto-pay includes my $350 monthly extra principal curtailment.",
            "I have $350 per month going toward extra mortgage principal automatically.",
            "Right now we send an extra $350 to principal on the mortgage each month.",
            "My automated extra principal payment on the house sits at $350 monthly.",
            "Each month our servicer pulls my $350 additional principal payment.",
            "I've kept our monthly extra mortgage principal transfer at $350.",
        ),
        ku_echo_replies_6=(
            "Checked the statement—servicer applied 100% to principal, not future interest.",
            "Love watching the principal balance drop faster each statement.",
            "Fits comfortably inside our post-tax cashflow.",
            "Great discipline alongside maxing retirement accounts.",
            "Escrow recalculation in autumn looks clean too.",
            "Once daycare drops off in September we'll have extra room!",
        ),
        ku_s9_update_claim="Now that preschool tuition ended in September, I increased our automatic monthly extra mortgage principal payment from $350 to $625.",
        ku_s9_reply="$625 a month cuts another two full years off the loan!",
        ku_question="What is Declan and Marisol's current automatic monthly extra mortgage principal payment?",
        ku_gold="$625 per month.",
        ku_stale_val="$350",
        attr_s6_turns=(
            ("Marisol", "Through my hospital benefits plan I'm contributing $3,400 to my individual HSA cuff plus employer seed."),
            ("Declan", "Since my tech plan covers the kids, I set my family-tier HSA payroll contribution to $4,900 invested in a total-market ETF."),
            ("Marisol", "Combined cleanly under the IRS family limit—good double check!"),
        ),
        attr_s8_distractor_turns=(
            ("Marisol", "Saved all our pediatric dental receipts digitally against my HSA archive folder."),
            ("Declan", "Tax-free compounding for decades."),
        ),
        attr_question="How much did Declan set for his HSA payroll contribution and how is it invested?",
        attr_gold="$4,900 invested in a total-market ETF.",
        abs_s10_turns=(
            ("Declan", "Reviewed beneficiaries and froze our credit files at Equifax, Experian, and TransUnion."),
            ("Marisol", "Annual financial housekeeping complete!"),
        ),
        abs_question="What variable interest rate margin did Declan negotiate on their Home Equity Line of Credit (HELOC)?",
        abs_gold_note="Unanswerable / not mentioned: a Home Equity Line of Credit (HELOC) is never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_16",
        domain="personal_finance",
        spk_a="Rohan",
        spk_b="Clara",
        needle_context_turns=(
            ("Clara", "Did the rooftop solar installer clarify the microinverter parts-and-labor warranty?"),
            ("Rohan", "Yes, read the Enphase contract fine print line by line."),
            ("Rohan", "The IQ8M microinverters carry a 25-year equipment warranty and our installer included a 12-year roof-penetration workmanship guarantee."),
            ("Clara", "25 years on the microinverters and 12 years on roof penetrations gives real peace of mind."),
        ),
        needle_question="What equipment warranty and roof-penetration workmanship guarantee durations did Rohan confirm for the solar installation?",
        needle_gold="A 25-year equipment warranty on the IQ8M microinverters and a 12-year roof-penetration workmanship guarantee.",
        synth_s1_turns=(
            ("Rohan", "Received our utility smart-thermostat & attic air-sealing rebate check in January for $900."),
            ("Clara", "Straight into the home-efficiency sinking fund!"),
        ),
        synth_s4_turns=(
            ("Rohan", "April brought the state clean-heating rebate wire of $2,400 for the cold-climate heat pump."),
            ("Clara", "That covered nearly a quarter of the condenser upgrade."),
        ),
        synth_s7_turns=(
            ("Rohan", "July municipal solar-interconnect incentive paid out $1,300 after passing final electrical inspection."),
            ("Clara", "Every single rebate check cleared without a hitch."),
        ),
        synth_question="What was the total dollar amount of the three energy rebates Rohan received in January, April, and July?",
        synth_gold="$4,600 ($900 + $2,400 + $1,300).",
        ku_s2_initial_claim="I set our automatic monthly 529 college savings contribution to $400 per child.",
        ku_s2_reply="$400 per child captures the full state income-tax deduction.",
        ku_echoes_6=(
            "Each child's 529 plan receives my $400 automatic monthly contribution.",
            "I have our 529 auto-debit configured at $400 per month per child.",
            "Right now we contribute $400 monthly into each kid's 529 college account.",
            "Our monthly 529 college savings draft sits at $400 per child.",
            "I kept the automatic 529 transfer at $400 a month for each child.",
            "Both 529 accounts are running on my $400 per child monthly schedule.",
        ),
        ku_echo_replies_6=(
            "Age-based target enrollment glidepath keeps rebalancing automatic.",
            "Grandparents loved seeing the gift-contribution link too.",
            "Zero state tax on the deductible portion is so nice.",
            "Quarterly statement fees dropped another basis point.",
            "Electric bills fell $180/month now that solar is exporting!",
            "Should we redirect our electric-bill savings into the 529s?",
        ),
        ku_s9_update_claim="Redirecting our monthly utility savings after solar PTO, I raised our automatic 529 contribution from $400 to $525 per child each month.",
        ku_s9_reply="Turning sunshine on the roof into an extra $125 per kid every month is poetic!",
        ku_question="How much is Rohan and Clara's current automatic monthly 529 contribution per child?",
        ku_gold="$525 per child per month.",
        ku_stale_val="$400",
        attr_s6_turns=(
            ("Clara", "In my Roth IRA at Fidelity I consolidated everything into FZROX with a 0.00% expense ratio."),
            ("Rohan", "Nice—in my rollover IRA at Vanguard I hold VTWAX global equity at a 0.10% expense ratio for automatic international weight."),
            ("Clara", "Both dirt-cheap index holdings with zero tinkering required."),
        ),
        attr_s8_distractor_turns=(
            ("Clara", "Reinvested third-quarter dividends inside my 0.00% FZROX Roth IRA."),
            ("Rohan", "Stay-the-course indexing wins again."),
        ),
        attr_question="Which fund ticker and expense ratio does Rohan hold in his rollover IRA at Vanguard?",
        attr_gold="VTWAX with a 0.10% expense ratio.",
        abs_s10_turns=(
            ("Rohan", "Exported 1.4 megawatt-hours net to the grid before autumn cloud cover rolled in."),
            ("Clara", "Credit rolls forward against January heating kWh!"),
        ),
        abs_question="How many kilowatt-hours of storage capacity did Rohan install with a Tesla Powerwall home battery?",
        abs_gold_note="Unanswerable / not mentioned: a Tesla Powerwall or home battery backup system was never installed or mentioned.",
        extra_closing_sessions=(
            (
                "2025-11-20 20:00",
                (
                    ("Clara", "Heat pump held 68F effortlessly during last night's 19F frost."),
                    ("Rohan", "Quiet outdoor unit too!"),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_17",
        domain="personal_finance",
        spk_a="Malcolm",
        spk_b="Esme",
        needle_context_turns=(
            ("Esme", "How much is the new boutique CPA firm charging for monthly studio bookkeeping and payroll filing?"),
            ("Malcolm", "Switched off our clunky DIY spreadsheet setup on Friday."),
            ("Malcolm", "Their flat monthly bookkeeping retainer is $380, which includes Gusto payroll reconciliation and state sales-tax filings."),
            ("Esme", "$380 a month saves us six hours of Sunday admin headaches easily."),
        ),
        needle_question="What is the flat monthly bookkeeping retainer fee from Malcolm and Esme's CPA firm?",
        needle_gold="$380 per month.",
        synth_s1_turns=(
            ("Malcolm", "Paid our Q1 federal estimated tax voucher via EFTPS for $6,200 after the healthcare branding project closed."),
            ("Esme", "Great start to studio billings."),
        ),
        synth_s4_turns=(
            ("Malcolm", "Submitted Q2 federal estimated tax payment of $7,800 following the fintech web-app retainer."),
            ("Esme", "Safe-harbor tracker is right on target."),
        ),
        synth_s7_turns=(
            ("Malcolm", "Remitted Q3 federal estimated taxes of $6,500 ahead of the September 15 deadline."),
            ("Esme", "Zero underpayment penalty stress come April."),
        ),
        synth_question="What was the combined dollar amount of the Q1, Q2, and Q3 federal estimated tax payments Malcolm remitted?",
        synth_gold="$20,500 ($6,200 in Q1 + $7,800 in Q2 + $6,500 in Q3).",
        ku_s2_initial_claim="I set our studio's high-yield operating cash reserve target at $30,000.",
        ku_s2_reply="$30,000 covers roughly four months of lean fixed overhead.",
        ku_echoes_6=(
            "Our studio operating cash reserve target in the high-yield account is $30,000.",
            "I keep our studio's high-yield operating reserve goal at $30,000.",
            "Right now our business operating reserve target sits at $30,000.",
            "We maintain a $30,000 target floor in the studio high-yield reserve.",
            "My target balance for the studio operating cash reserve remains $30,000.",
            "The operating reserve buffer I set for the studio is $30,000.",
        ),
        ku_echo_replies_6=(
            "Earning 4.4% APY on operating float feels great.",
            "Lets us turn down bad-fit rush projects calmly.",
            "CPA liked seeing clean separation from tax-withholding buckets.",
            "Owner distributions above the cap stay super predictable.",
            "Did bringing on the part-time motion designer raise monthly burn?",
            "Yeah, fixed commitments ticked up once we signed the studio loft renewal.",
        ),
        ku_s9_update_claim="After renewing our studio lease and adding part-time contractor hours, I raised our operating cash reserve target from $30,000 to $45,000.",
        ku_s9_reply="$45,000 preserves a full six-month runway at our new overhead level.",
        ku_question="What is the studio's current high-yield operating cash reserve target?",
        ku_gold="$45,000.",
        ku_stale_val="$30,000",
        attr_s6_turns=(
            ("Esme", "For my color-grading desk I expensed an Asus ProArt 32-inch OLED reference monitor at $1,290."),
            ("Malcolm", "And for my workstation I bought a Herman Miller Aeron Remastered chair and walnut standing desk totaling $1,840."),
            ("Esme", "Section 179 equipment budget put to awesome ergonomic use!"),
        ),
        attr_s8_distractor_turns=(
            ("Esme", "Hardware calibration puck confirmed Delta-E under 1.0 on my $1,290 ProArt OLED."),
            ("Malcolm", "Print proofs matched screen colors on the very first press run."),
        ),
        attr_question="What workstation equipment did Malcolm buy for himself and what was the total cost?",
        attr_gold="A Herman Miller Aeron Remastered chair and walnut standing desk totaling $1,840.",
        abs_s10_turns=(
            ("Malcolm", "Opened our Solo 401(k) profit-sharing sub-accounts before the December 31 plan deadline."),
            ("Esme", "Best year our independent studio has ever had."),
        ),
        abs_question="What business-use mileage percentage did Malcolm claim on an S-corp electric vehicle lease?",
        abs_gold_note="Unanswerable / not mentioned: an electric vehicle lease or mileage deduction is never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_18",
        domain="personal_finance",
        spk_a="Oren",
        spk_b="Beatrix",
        needle_context_turns=(
            ("Beatrix", "Did the environmental test on the duplex basement come back below the EPA 4.0 pCi/L action threshold?"),
            ("Oren", "Came back at 5.6 pCi/L, so we went ahead and mitigated proactively before listing Unit B."),
            ("Oren", "The certified sub-slab radon mitigation system with continuous digital manometer cost $1,480 installed."),
            ("Beatrix", "Follow-up 48-hour charcoal test dropped to 0.7 pCi/L—$1,480 very well spent."),
        ),
        needle_question="How much did Oren pay to install the sub-slab radon mitigation system?",
        needle_gold="$1,480.",
        synth_s1_turns=(
            ("Oren", "Paid Contractor Draw #1 of $8,500 in January after rough plumbing and PEX repiping passed city inspection."),
            ("Beatrix", "Goodbye 1960s galvanized supply lines!"),
        ),
        synth_s4_turns=(
            ("Oren", "Cleared Contractor Draw #2 of $11,200 in April once LVP flooring and acoustic subfloor underlayment wrapped."),
            ("Beatrix", "Footstep noise between upstairs and downstairs vanished."),
        ),
        synth_s7_turns=(
            ("Oren", "Released final Contractor Draw #3 of $6,300 in July after quartz countertops and punch-list sign-off."),
            ("Beatrix", "Unit B leased on the first open-house weekend!"),
        ),
        synth_question="What was the combined dollar total across Contractor Draws #1, #2, and #3 that Oren paid?",
        synth_gold="$26,000 ($8,500 + $11,200 + $6,300).",
        ku_s2_initial_claim="We currently carry a $1,000,000 personal umbrella liability policy over our home and auto.",
        ku_s2_reply="$1,000,000 umbrella was fine before we owned rental units.",
        ku_echoes_6=(
            "Our personal umbrella liability coverage limit stands at $1,000,000.",
            "Right now we have a $1,000,000 umbrella policy layered above home and auto.",
            "I still carry our $1,000,000 personal umbrella liability limit.",
            "Our existing umbrella insurance policy provides $1,000,000 in coverage.",
            "We're covered by a $1,000,000 personal umbrella liability policy.",
            "The coverage cap on our personal umbrella policy is $1,000,000.",
        ),
        ku_echo_replies_6=(
            "Make sure the broker lists both Unit A and Unit B on the schedule.",
            "Underlying landlord liability needs $300k minimum per occurrence.",
            "Premium per million is surprisingly cheap.",
            "Let's review limits once tenants move into Unit B.",
            "Independent broker is quoting Chubb and RLI this week.",
            "Definitely worth bumping up now that rental lease is signed.",
        ),
        ku_s9_update_claim="With both duplex units occupied, I increased our personal umbrella liability policy coverage limit from $1,000,000 to $2,000,000.",
        ku_s9_reply="Doubling to $2,000,000 only added $190 a year—total no-brainer.",
        ku_question="What is the current coverage limit on Oren and Beatrix's personal umbrella liability policy?",
        ku_gold="$2,000,000.",
        ku_stale_val="$1,000,000",
        attr_s6_turns=(
            ("Beatrix", "Put the appliance package on my Chase Ink Business Cash card and earned a 75,000-point sign-up bonus."),
            ("Oren", "Sweet—putting lumber and tile materials on my Amex Blue Business Plus netted me a 50,000-point welcome bonus."),
            ("Beatrix", "Renovation spend funding next spring's flights for free!"),
        ),
        attr_s8_distractor_turns=(
            ("Beatrix", "My 75,000 Chase Ink bonus points posted cleanly on this month's statement."),
            ("Oren", "Zero interest carried on either card too."),
        ),
        attr_question="Which credit card and welcome bonus did Oren earn from renovation materials spend?",
        attr_gold="Amex Blue Business Plus with a 50,000-point welcome bonus.",
        abs_s10_turns=(
            ("Oren", "Set up separate security-deposit escrow sub-accounts for both long-term annual tenants."),
            ("Beatrix", "Smooth, boring long-term tenancy is the goal."),
        ),
        abs_question="How much was the city application fee for Oren's short-term Airbnb vacation rental permit?",
        abs_gold_note="Unanswerable / not mentioned: they leased to long-term annual tenants and never applied for or mentioned a short-term Airbnb permit.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_19",
        domain="personal_finance",
        spk_a="Cedric",
        spk_b="Viveka",
        needle_context_turns=(
            ("Viveka", "Did Schwab charge an outgoing international wire fee when we sent the Lisbon apartment deposit?"),
            ("Cedric", "Used Wise instead of a traditional SWIFT bank wire so we didn't get hit on FX spread."),
            ("Cedric", "Total conversion and SEPA transfer fee on the 6,000-euro transfer came to exactly $26.40."),
            ("Viveka", "$26.40 mid-market beats a 3% bank FX markup by over a hundred fifty bucks."),
        ),
        needle_question="What was the exact fee in dollars for Cedric's 6,000-euro Wise conversion and SEPA transfer?",
        needle_gold="$26.40.",
        synth_s1_turns=(
            ("Cedric", "Our 6-month credit-union CD matured in January, paying out $10,400 into our sabbatical bucket."),
            ("Viveka", "Bucket one funded!"),
        ),
        synth_s4_turns=(
            ("Cedric", "Second CD matured in April and deposited $14,600 into the sabbatical cash account."),
            ("Viveka", "Over halfway to our full travel cash cushion."),
        ),
        synth_s7_turns=(
            ("Cedric", "Final 12-month CD matured in July for $12,000 flat."),
            ("Viveka", "Sabbatical cash runway is 100% liquid and ready!"),
        ),
        synth_question="What was the combined dollar payout across the three maturing CDs in January, April, and July?",
        synth_gold="$37,000 ($10,400 + $14,600 + $12,000).",
        ku_s2_initial_claim="My Investment Policy Statement sets our international developed-and-emerging equity allocation at 25% of equities.",
        ku_s2_reply="25% international has been easy to maintain across tax-advantaged accounts.",
        ku_echoes_6=(
            "Our portfolio IPS holds international equities at 25% of our stock allocation.",
            "I keep our target international equity weight at 25% of total equities.",
            "Right now international stocks represent my 25% target slice of equities.",
            "We rebalance international developed-and-emerging stocks to 25% of equities.",
            "My IPS target for international equity allocation remains 25%.",
            "International equity sits at a 25% target share in our portfolio IPS.",
        ),
        ku_echo_replies_6=(
            "Foreign tax credit in taxable covers a nice chunk of withholding.",
            "VXUS + AVDV tilts look super clean.",
            "Rebalancing bands at +/-3% prevented unnecessary trades.",
            "Living abroad six months made us think about USD concentration.",
            "Global market-cap weight is closer to 35-38% anyway.",
            "Annual IPS review is next month—let's update if we agree.",
        ),
        ku_s9_update_claim="During our September IPS review ahead of living in Europe, I raised our target international equity allocation from 25% to 35% of equities.",
        ku_s9_reply="35% international aligns way better with global market cap and multi-currency spending.",
        ku_question="What is Cedric and Viveka's current target international equity allocation percentage in their IPS?",
        ku_gold="35% of equities.",
        ku_stale_val="25%",
        attr_s6_turns=(
            ("Viveka", "For my standalone dental policy during sabbatical I picked Delta Dental PPO with a $2,000 annual maximum."),
            ("Cedric", "And for our GeoBlue Voyager international medical policy I selected a $500 annual deductible with $1M evacuation coverage."),
            ("Viveka", "$500 deductible on GeoBlue plus $2,000 dental max leaves zero healthcare gaps."),
        ),
        attr_s8_distractor_turns=(
            ("Viveka", "Confirmed routine six-month cleanings have zero waiting period on my $2,000 Delta Dental PPO."),
            ("Cedric", "GeoBlue physical ID cards arrived in the mail yesterday."),
        ),
        attr_question="Which international medical policy and annual deductible did Cedric select?",
        attr_gold="GeoBlue Voyager with a $500 annual deductible (and $1M evacuation coverage).",
        abs_s10_turns=(
            ("Cedric", "Set up mail scanning and two YubiKey hardware keys (one worn, one locked in safe) before departure."),
            ("Viveka", "Six-month sabbatical countdown is down to three weeks!"),
        ),
        abs_question="How many months of interest penalty did Cedric forfeit when redeeming US Treasury Series I savings bonds?",
        abs_gold_note="Unanswerable / not mentioned: Series I savings bonds are never mentioned in any session.",
        extra_closing_sessions=(
            (
                "2025-11-15 12:30",
                (
                    ("Viveka", "Landed in Lisbon and apartment keys worked on the first turn!"),
                    ("Cedric", "Pastéis de nata on the balcony—we made it!"),
                ),
            ),
        ),
    ),
    # =========================================================================
    # DOMAIN 5: OPERATIONS (4 conversations: vconv_20 .. vconv_23)
    # =========================================================================
    _ScenarioSpec(
        conv_id="vconv_20",
        domain="operations",
        spk_a="Gareth",
        spk_b="Farah",
        needle_context_turns=(
            ("Farah", "Did NIST traceable recalibration pass on all twelve ultra-low biologics freezers in Vault C?"),
            ("Gareth", "Eleven passed within +/-0.1C; Probe #7 drifted slightly so we entered a calibration offset."),
            ("Gareth", "Glycol bottle sensor `PT100-C07` showed a +0.34C offset at -80C and was replaced with spare probe lot `NIST-8841`."),
            ("Farah", "Logging lot `NIST-8841` in the GMP deviation register keeps QA audit-ready."),
        ),
        needle_question="Which spare probe lot number replaced glycol bottle sensor `PT100-C07`, and what offset had it shown?",
        needle_gold="Spare probe lot `NIST-8841` (replacing `PT100-C07` after a +0.34C offset at -80C).",
        synth_s1_turns=(
            ("Gareth", "January dry-ice pellet audit: Hub North consumed 1,600 kilograms for phase-III clinical shipper validation."),
            ("Farah", "Winter ambient temperatures helped hold sublimation down."),
        ),
        synth_s4_turns=(
            ("Gareth", "April cold-chain dispatch used 2,100 kilograms of dry-ice pellets across Hub Central's oncology lane."),
            ("Farah", "Zero thermal excursions across 420 validated shippers."),
        ),
        synth_s7_turns=(
            ("Gareth", "July peak summer heat pushed Hub South dry-ice pellet usage to 2,700 kilograms."),
            ("Farah", "VIP vacuum-insulated shippers earned every penny in July."),
        ),
        synth_question="How many total kilograms of dry-ice pellets were consumed across the January (Hub North), April (Hub Central), and July (Hub South) reports?",
        synth_gold="6,400 kilograms (1,600 kg + 2,100 kg + 2,700 kg).",
        ku_s2_initial_claim="Our -80C biologics freezer door-open alarm delay in Vault C is set to 90 seconds.",
        ku_s2_reply="90 seconds prevents nuisance pages when techs pull single cryoboxes.",
        ku_echoes_6=(
            "Vault C's -80C freezer door-ajar alarm delay is configured at 90 seconds.",
            "I have the door-open alarm delay on our Vault C -80C freezers set to 90 seconds.",
            "Right now Vault C -80C units wait 90 seconds before firing a door-open alarm.",
            "Our GMP alarm SOP holds Vault C -80C door-open delay at 90 seconds.",
            "I kept the -80C freezer door-open alarm delay in Vault C at 90 seconds.",
            "Vault C's ultra-low freezers operate with a 90-second door-open alarm delay.",
        ),
        ku_echo_replies_6=(
            "Inner insulated sub-doors hold chamber air steady.",
            "Good, night shift logged zero false alarms this month.",
            "Matches our building management SCADA escalation tree.",
            "EU Annex 11 auditor asked about high-value mRNA racks yesterday.",
            "Yeah, the new mRNA lot has tighter transient warm-up tolerances.",
            "Change control CC-209 is approved whenever you're ready to tighten it.",
        ),
        ku_s9_update_claim="Under approved change control CC-209 for the mRNA lot, I tightened Vault C's -80C freezer door-open alarm delay from 90 seconds down to 45 seconds.",
        ku_s9_reply="45 seconds passed OQ alarm verification on all twelve Vault C chambers today.",
        ku_question="What is the current -80C freezer door-open alarm delay in Vault C?",
        ku_gold="45 seconds.",
        ku_stale_val="90",
        attr_s6_turns=(
            ("Farah", "I took ownership of the 2-to-8C refrigerated walk-in Room B2 weekly LN2 backup valve inspection."),
            ("Gareth", "And I'm primary owner of the -196C vapor-phase cryo-tank auto-fill manifold sign-off in Bay D4."),
            ("Farah", "Clear zone split on the SOP matrix—Room B2 on me, Bay D4 cryo manifold on you."),
        ),
        attr_s8_distractor_turns=(
            ("Farah", "Completed my Room B2 weekly backup valve inspection ahead of Friday's client tour."),
            ("Gareth", "QA signed off both binders without a single observation."),
        ),
        attr_question="Which inspection sign-off and bay is Gareth primary owner of?",
        attr_gold="The -196C vapor-phase cryo-tank auto-fill manifold sign-off in Bay D4.",
        abs_s10_turns=(
            ("Gareth", "Annual 24-hour diesel generator load-bank failover test succeeded with seamless ATS transfer."),
            ("Farah", "Ready for winter storm season."),
        ),
        abs_question="How many hours was clinical shipment #409 held by FDA customs inspectors at the Memphis air-freight hub?",
        abs_gold_note="Unanswerable / not mentioned: neither Memphis air-freight nor shipment #409 or an FDA customs hold is ever mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_21",
        domain="operations",
        spk_a="Torben",
        spk_b="Keiko",
        needle_context_turns=(
            ("Keiko", "Why were three autonomous mobile robots throwing false obstacle halts near shrink-wrap stations?"),
            ("Torben", "Static dust film on the safety scanner lenses scattered the infrared return beam."),
            ("Torben", "Added an anti-static IPA microfiber wipe of every SICK microScan3 LiDAR window every 72 operating hours."),
            ("Keiko", "72-hour lens wipe PM eliminated false e-stops completely this week."),
        ),
        needle_question="Every how many operating hours did Torben schedule cleaning of the SICK microScan3 LiDAR windows?",
        needle_gold="Every 72 operating hours.",
        synth_s1_turns=(
            ("Torben", "January one-way aisle traffic routing added 140 extra pallets per shift in Zone A."),
            ("Keiko", "Deadlocks at the main cross-aisle vanished overnight."),
        ),
        synth_s4_turns=(
            ("Torben", "April dynamic slotting of fast-moving SKUs gained another 195 pallets per shift in Zone B."),
            ("Keiko", "Travel distance per pick dropped 22%."),
        ),
        synth_s7_turns=(
            ("Torben", "July dual-tray tote induction added 165 pallets per shift in Zone C."),
            ("Keiko", "Outbound dock doors are turning trailers faster than ever."),
        ),
        synth_question="Across the January (Zone A), April (Zone B), and July (Zone C) improvements, how many total extra pallets per shift did Torben add?",
        synth_gold="500 extra pallets per shift (140 + 195 + 165).",
        ku_s2_initial_claim="I set our AMR fleet opportunity-charging trigger at 28% battery state-of-charge.",
        ku_s2_reply="28% SoC protects LFP cell cycle life nicely.",
        ku_echoes_6=(
            "Our AMR robots head to opportunity-charging pads when battery SoC hits 28%.",
            "I have the AMR fleet low-battery charging threshold set at 28% SoC.",
            "Right now warehouse AMRs trigger opportunity charging at 28% state-of-charge.",
            "My fleet manager rule dispatches AMRs to chargers at 28% SoC.",
            "I've kept the AMR opportunity-charge floor at 28% battery SoC.",
            "Fleet robots still break for opportunity charging at the 28% SoC mark.",
        ),
        ku_echo_replies_6=(
            "Eight fast-charge pads in the east mezzanine keep queues zero.",
            "Cell balancing happens automatically on Sunday night full charges.",
            "Good, zero stranded bots in narrow racking aisles.",
            "Peak Q4 double-shift simulations showed a small charger bottleneck at shift change.",
            "Raising Opportunity SoC slightly could stagger dock arrivals earlier.",
            "Let's test a higher trigger before October peak volume.",
        ),
        ku_s9_update_claim="To prevent shift-change charger queuing ahead of Q4 peak, I raised the AMR fleet opportunity-charging trigger from 28% to 36% battery state-of-charge.",
        ku_s9_reply="36% SoC staggered short 8-minute top-ups across the whole afternoon—zero charger wait!",
        ku_question="At what battery state-of-charge percentage does Torben's AMR fleet currently trigger opportunity charging?",
        ku_gold="36% state-of-charge.",
        ku_stale_val="28%",
        attr_s6_turns=(
            ("Keiko", "I'm handling the Siemens S7-1500 sortation conveyor PLC firmware patching during Sunday's maintenance window."),
            ("Torben", "Great—I'll own upgrading the Cognex DataMan overhead tunnel barcode readers to firmware v6.4.2."),
            ("Keiko", "Conveyor PLC on me and Cognex tunnel scanners on you—we'll be done by 03:00."),
        ),
        attr_s8_distractor_turns=(
            ("Keiko", "Siemens S7-1500 conveyor PLC patch ran cleanly with zero Profinet drops."),
            ("Torben", "No-read scan rate fell below 0.08% after the Cognex update too."),
        ),
        attr_question="Which hardware system and firmware version upgrade did Torben own during the maintenance window?",
        attr_gold="Upgrading the Cognex DataMan overhead tunnel barcode readers to firmware v6.4.2.",
        abs_s10_turns=(
            ("Torben", "Peak readiness drill hit 99.4% on-time parcel cutoffs for regional carriers."),
            ("Keiko", "Bring on November cyber week!"),
        ),
        abs_question="Which vendor did Torben select for the hands-free voice-picking Bluetooth headsets?",
        abs_gold_note="Unanswerable / not mentioned: voice-picking headsets are never mentioned anywhere in the conversation.",
        extra_closing_sessions=(
            (
                "2025-11-22 07:45",
                (
                    ("Keiko", "Record 18,400 cartons shipped on Friday night shift with zero safety incidents."),
                    ("Torben", "Crew earned Monday catered lunch big time!"),
                ),
            ),
        ),
    ),
    _ScenarioSpec(
        conv_id="vconv_22",
        domain="operations",
        spk_a="Damian",
        spk_b="Noor",
        needle_context_turns=(
            ("Noor", "Did we validate how long thawed enzymatic fragmentation master-mix stays active on the cold block?"),
            ("Damian", "Ran duplicate QC libraries across four timepoints on TapeStation."),
            ("Damian", "Fragment size distribution stayed within +/-8 bp up to 4.5 hours at 4C using master-mix lot `EFX-419B`."),
            ("Noor", "Setting a strict 4-hour benchtop expiry on lot `EFX-419B` leaves a clean safety margin."),
        ),
        needle_question="Which enzymatic fragmentation master-mix lot number did Damian validate, and up to how many hours at 4C did fragment size stay within +/-8 bp?",
        needle_gold="Lot `EFX-419B`, validated up to 4.5 hours at 4C.",
        synth_s1_turns=(
            ("Damian", "January sequencing throughput: completed 14 NovaSeq X Plus 10B flowcell runs for the oncology cohort."),
            ("Noor", "Q30 base quality averaged 93.8% across the board."),
        ),
        synth_s4_turns=(
            ("Damian", "April core-lab summary: wrapped 19 NovaSeq X Plus 10B flowcell runs as rare-disease trios ramped up."),
            ("Noor", "Turnaround time stayed under six business days."),
        ),
        synth_s7_turns=(
            ("Damian", "July clinical surge completed 21 NovaSeq X Plus 10B flowcell runs with zero failed lanes."),
            ("Noor", "Zero repeat runs all month saved huge reagent budget."),
        ),
        synth_question="How many total NovaSeq X Plus 10B flowcell runs did Damian complete across January, April, and July?",
        synth_gold="54 flowcell runs (14 in January + 19 in April + 21 in July).",
        ku_s2_initial_claim="I calibrated our final NovaSeq library loading concentration to 165 picomolar.",
        ku_s2_reply="165 picomolar gives healthy cluster occupancy without polyclonal overclustering.",
        ku_echoes_6=(
            "Our final NovaSeq library loading concentration is set at 165 picomolar.",
            "I load pooled clinical libraries onto NovaSeq flowcells at 165 picomolar.",
            "Right now our SOP uses my 165-picomolar final loading concentration.",
            "Final denatured pool loading molarity in the core lab sits at 165 picomolar.",
            "I've kept our NovaSeq 10B final loading target at 165 picomolar.",
            "Pooled sequencing runs still use the 165-picomolar loading concentration.",
        ),
        ku_echo_replies_6=(
            "%PF passing filter stayed right around 84%.",
            "Index hopping rate below 0.02% with unique dual indexes.",
            "Good, qPCR KAPA quantification matches Qubit nicely.",
            "Illumina's v1.3 XLEAP-SBS chemistry kit arrived for September.",
            "Right, XLEAP v1.3 prefers slightly lower loading molarity to maximize %PF.",
            "Let's run the titration strip before locking the new SOP value.",
        ),
        ku_s9_update_claim="After validating Illumina's v1.3 XLEAP-SBS chemistry titration, I lowered our final NovaSeq library loading concentration from 165 picomolar to 135 picomolar.",
        ku_s9_reply="135 picomolar pushed %PF passing filter up to 89.4% with crystal-clear Q30 scores!",
        ku_question="What is the core lab's current final NovaSeq library loading concentration in picomolar?",
        ku_gold="135 picomolar.",
        ku_stale_val="165",
        attr_s6_turns=(
            ("Noor", "For quarterly CAP proficiency readiness, I signed up to run gravimetric channel verification on our Hamilton STARlet liquid handler."),
            ("Damian", "And I'll run the optical LED standard verification across all four Qubit Flex fluorometers and Bioanalyzer chips."),
            ("Noor", "Hamilton gravimetrics on me and Qubit Flex optical standards on you—CAP binder will be spotless."),
        ),
        attr_s8_distractor_turns=(
            ("Noor", "All eight channels on the Hamilton STARlet passed gravimetric CV under 1.2% at 5 microliters."),
            ("Damian", "Qubit Flex standards hit R-squared 0.9998 too."),
        ),
        attr_question="Which instruments did Damian verify for quarterly CAP proficiency readiness?",
        attr_gold="All four Qubit Flex fluorometers (optical LED standards) and Bioanalyzer chips.",
        abs_s10_turns=(
            ("Damian", "Passed our unannounced CAP clinical genomics inspection with zero deficiencies!"),
            ("Noor", "Best possible outcome for the whole lab team."),
        ),
        abs_question="What N50 read length did Damian achieve on the Oxford Nanopore PromethION long-read run?",
        abs_gold_note="Unanswerable / not mentioned: Oxford Nanopore and PromethION long-read sequencing are never mentioned.",
        extra_closing_sessions=(),
    ),
    _ScenarioSpec(
        conv_id="vconv_23",
        domain="operations",
        spk_a="Henning",
        spk_b="Valeria",
        needle_context_turns=(
            ("Valeria", "Did the utility protection relay pass power-quality commissioning on Feeder 2?"),
            ("Henning", "Captured 7-day Fluke 435 waveform logs across full solar noon clipping."),
            ("Henning", "Total harmonic distortion (THD) on the grid-forming SMA Sunny Central inverter measured 1.4% voltage THD, well below the IEEE 519 5.0% limit."),
            ("Valeria", "1.4% voltage THD keeps the island desalination variable-frequency drives super happy."),
        ),
        needle_question="What voltage total harmonic distortion (THD) percentage did Henning measure on the grid-forming SMA Sunny Central inverter?",
        needle_gold="1.4% voltage THD.",
        synth_s1_turns=(
            ("Henning", "January nor'easter microgrid islanding report: battery + PV dispatch avoided 38 hours of diesel generator runtime."),
            ("Valeria", "Nearly 4,000 liters of barged marine diesel saved in one storm."),
        ),
        synth_s4_turns=(
            ("Henning", "April subsea-cable maintenance outage: microgrid carried the harbor load and avoided 52 hours of diesel generator runtime."),
            ("Valeria", "Town hospital didn't even see a voltage flicker during seamless island transition."),
        ),
        synth_s7_turns=(
            ("Henning", "July tropical squall line outage: solar-plus-storage avoided 30 hours of diesel runtime across the weekend."),
            ("Valeria", "Quiet, zero-exhaust harbor nights all weekend."),
        ),
        synth_question="Across the January, April, and July outage events, how many total hours of diesel generator runtime did the microgrid avoid?",
        synth_gold="120 hours (38 hours + 52 hours + 30 hours).",
        ku_s2_initial_claim="I set our evening peak-shaving BESS state-of-charge reserve floor at 22%.",
        ku_s2_reply="22% SoC reserve leaves enough black-start energy for summer evenings.",
        ku_echoes_6=(
            "Our BESS evening peak-shaving state-of-charge reserve floor sits at 22%.",
            "I have the microgrid battery reserve floor set to 22% SoC during evening peak shaving.",
            "Right now our dispatch controller holds a 22% BESS state-of-charge reserve floor.",
            "Evening peak-shaving stops discharging once the battery reaches my 22% SoC floor.",
            "I've kept the microgrid BESS reserve floor configured at 22% state-of-charge.",
            "The battery energy storage system operates with a 22% evening SoC reserve floor.",
        ),
        ku_echo_replies_6=(
            "Round-trip AC-AC efficiency held 87.5% this month.",
            "HVAC chiller staged cycling keeps rack cells right at 24C.",
            "Good, evening ferry shore-power plug-ins absorbed smoothly.",
            "Autumn hurricane watch season starts mid-September though.",
            "Higher storm outage buffer overnight makes total sense in autumn.",
            "Let's raise the reserve floor before September equinox gales.",
        ),
        ku_s9_update_claim="Entering autumn storm season, I raised our evening peak-shaving BESS state-of-charge reserve floor from 22% to 40%.",
        ku_s9_reply="Holding a 40% SoC floor guarantees six hours of critical hospital and water-plant autonomy overnight.",
        ku_question="What is the microgrid's current evening peak-shaving BESS state-of-charge reserve floor?",
        ku_gold="40%.",
        ku_stale_val="22%",
        attr_s6_turns=(
            ("Valeria", "I reconfigured our Campbell Scientific irradiance and anemometer weather station to publish MQTT packets every 2 seconds."),
            ("Henning", "Nice—on the SEL-3555 RTAC controller I tuned Modbus TCP register polling across the battery racks to 250 milliseconds."),
            ("Valeria", "250-millisecond rack Modbus polling plus 2-second sky-cam irradiance forecasts gives super tight ramp control."),
        ),
        attr_s8_distractor_turns=(
            ("Valeria", "Cloud-edge PV ramp forecasts improved 18% with my 2-second weather station MQTT stream."),
            ("Henning", "Frequency stayed inside 59.98 to 60.02 Hz all afternoon."),
        ),
        attr_question="What Modbus TCP polling interval did Henning configure on the SEL-3555 RTAC controller?",
        attr_gold="250 milliseconds.",
        abs_s10_turns=(
            ("Henning", "Completed infrared thermography across all combiner boxes and medium-voltage switchgear—zero hot spots."),
            ("Valeria", "Microgrid availability for Year 1 closed at 99.98%."),
        ),
        abs_question="How many kilowatts did the subsea tidal-stream turbine contribute during spring tides?",
        abs_gold_note="Unanswerable / not mentioned: a subsea tidal-stream turbine is never mentioned in any session.",
        extra_closing_sessions=(),
    ),
)


def _filler_opening(domain: str, spk_a: str, spk_b: str, sess_idx: int, conv_idx: int) -> tuple[tuple[str, str], ...]:
    """Deterministic natural session icebreakers by domain so sessions read like realistic multi-turn logs."""
    openers = {
        "software_engineering": (
            ("Quick sync before standup—CI green on main after last night's merge?", "Yep, all unit and integration suites passed in seven minutes."),
            ("Grabbed coffee—ready to walk through our infra notes for this sprint?", "Ready! Pulled up the service dashboard right here."),
            ("Checking in mid-sprint—any alerts fire during the EU traffic window?", "Quiet pager all morning, error budget is looking healthy."),
        ),
        "health_fitness": (
            ("Morning! How are the legs feeling after yesterday's recovery spin?", "Way looser today, eight hours of sleep worked wonders."),
            ("Quick training log check-in before dinner—hydration holding up this week?", "Feeling super steady, morning energy has been crisp."),
            ("Happy Friday! Ready to compare notes from this week's training block?", "Definitely, just uploaded my watch file."),
        ),
        "travel": (
            ("Pulled up our shared trip map—have five minutes to go over bookings?", "Always! Best part of the week is trip planning."),
            ("Quick itinerary check-in over tea—passports and confirmations all in one folder?", "All synced offline in the travel folder."),
            ("Checking exchange rates and weather norms for our route today!", "Every time I look at photos I get more excited."),
        ),
        "personal_finance": (
            ("Monthly money check-in—statements downloaded and budget sheet open?", "Right here with coffee, everything reconciled cleanly."),
            ("Quick household finance review before the weekend—any surprise bills?", "Zero surprises, utilities even came in slightly under budget."),
            ("Pulled up our brokerage and savings dashboard for our quarterly check!", "Love how calm these check-ins feel with automation running."),
        ),
        "operations": (
            ("Shift handoff check—all safety interlocks and environmental logs green?", "Green across the board, night crew left clean notes."),
            ("Mid-month ops sync—preventive maintenance tickets tracking on schedule?", "100% on time, zero overdue calibrations."),
            ("Quick walkthrough before the weekly ops huddle—inventory buffers healthy?", "Consumables and spares are well above reorder points."),
        ),
    }[domain]
    pair = openers[(sess_idx + conv_idx) % len(openers)]
    if sess_idx % 2 == 0:
        return ((spk_a, pair[0]), (spk_b, pair[1]))
    return ((spk_b, pair[0]), (spk_a, pair[1]))


def _build_conversation(conv_idx: int, spec: _ScenarioSpec) -> VirginConversation:
    if spec.spk_a.casefold() in LOCOMO_FORBIDDEN_SPEAKERS or spec.spk_b.casefold() in LOCOMO_FORBIDDEN_SPEAKERS:
        raise ValueError(f"Speaker overlap with LoCoMo detected in {spec.conv_id}")

    # Assemble 10..12 dated sessions per conversation (strictly inside 8..12!)
    session_payloads: list[tuple[str, str, list[tuple[str, str, str]]]] = []
    # Each entry in turn list: `(speaker, claim_text, role_tag)`

    # Session 1 (Jan): Synthesis Part 1
    s1_turns = [(sp, txt, "filler") for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, 0, conv_idx)]
    s1_turns.extend((sp, txt, "synth_1" if k == 0 else "synth_ctx") for k, (sp, txt) in enumerate(spec.synth_s1_turns))
    session_payloads.append((f"{spec.conv_id}_s01", _BASE_DATES_10[0], s1_turns))

    # Session 2 (Feb): Initial KU claim (`ku_old`)
    s2_turns = [(sp, txt, "filler") for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, 1, conv_idx)]
    s2_turns.append((spec.spk_a, spec.ku_s2_initial_claim, "ku_initial"))
    s2_turns.append((spec.spk_b, spec.ku_s2_reply, "ku_ctx"))
    session_payloads.append((f"{spec.conv_id}_s02", _BASE_DATES_10[1], s2_turns))

    # Sessions 3..8 (Mar..Aug): 6 non-Nemotron natural paraphrase echoes + interleaved needle/synthesis/attribution
    for echo_i in range(6):
        s_num = echo_i + 3  # 3..8
        dt_str = _BASE_DATES_10[s_num - 1]
        st: list[tuple[str, str, str]] = [
            (sp, txt, "filler")
            for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, s_num - 1, conv_idx)
        ]
        # Inject non-Nemotron natural paraphrase echo by `spk_a`
        st.append((spec.spk_a, spec.ku_echoes_6[echo_i], f"ku_echo_{echo_i}"))
        st.append((spec.spk_b, spec.ku_echo_replies_6[echo_i], "ku_echo_reply"))

        if s_num == 4:
            st.extend((sp, txt, "synth_2" if k == 0 else "synth_ctx") for k, (sp, txt) in enumerate(spec.synth_s4_turns))
        elif s_num == 5:
            for k, (sp, txt) in enumerate(spec.needle_context_turns):
                tag = "needle_target" if k == 2 else "needle_ctx"
                st.append((sp, txt, tag))
        elif s_num == 6:
            for k, (sp, txt) in enumerate(spec.attr_s6_turns):
                tag = "attr_target" if k == 1 else "attr_ctx"
                st.append((sp, txt, tag))
        elif s_num == 7:
            st.extend((sp, txt, "synth_3" if k == 0 else "synth_ctx") for k, (sp, txt) in enumerate(spec.synth_s7_turns))
        elif s_num == 8:
            st.extend((sp, txt, "attr_distractor") for sp, txt in spec.attr_s8_distractor_turns)

        session_payloads.append((f"{spec.conv_id}_s{s_num:02d}", dt_str, st))

    # Session 9 (Sep): Genuine KU update (`ku_new`)
    s9_turns = [(sp, txt, "filler") for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, 8, conv_idx)]
    s9_turns.append((spec.spk_a, spec.ku_s9_update_claim, "ku_update"))
    s9_turns.append((spec.spk_b, spec.ku_s9_reply, "ku_update_reply"))
    session_payloads.append((f"{spec.conv_id}_s09", _BASE_DATES_10[8], s9_turns))

    # Session 10 (Oct): Post-update wrap-up (ensuring KU update is NOT the final session)
    s10_turns = [(sp, txt, "filler") for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, 9, conv_idx)]
    s10_turns.extend((sp, txt, "abs_neighborhood") for sp, txt in spec.abs_s10_turns)
    session_payloads.append((f"{spec.conv_id}_s10", _BASE_DATES_10[9], s10_turns))

    # Optional sessions 11..12 so session count spans 10..12 across the 24 conversations
    for extra_i, (extra_dt, extra_raw) in enumerate(spec.extra_closing_sessions, start=11):
        ext_turns = [(sp, txt, "filler") for sp, txt in _filler_opening(spec.domain, spec.spk_a, spec.spk_b, extra_i, conv_idx)]
        ext_turns.extend((sp, txt, "closing") for sp, txt in extra_raw)
        session_payloads.append((f"{spec.conv_id}_s{extra_i:02d}", extra_dt, ext_turns))

    if not (8 <= len(session_payloads) <= 12):
        raise ValueError(f"{spec.conv_id}: expected 8..12 sessions, got {len(session_payloads)}")

    turns: list[ConversationTurn] = []
    tagged_turn_ids: dict[str, str] = {}
    tagged_session_ids: dict[str, str] = {}

    for sid, dt_str, raw_turns in session_payloads:
        base_ts = _parse_utc_ts(dt_str)
        for pos, (spk, claim, tag) in enumerate(raw_turns):
            tid = f"{sid}_t{pos:02d}"
            turns.append(
                ConversationTurn(
                    turn_id=tid,
                    session_id=sid,
                    date=dt_str,
                    timestamp=base_ts + pos * 1.0,
                    text=f"{spk}: {claim}",
                    claim_text=claim,
                    speaker=spk,
                )
            )
            if tag in {"needle_target", "synth_1", "synth_2", "synth_3", "ku_initial", "ku_update", "attr_target"}:
                tagged_turn_ids[tag] = tid
                tagged_session_ids[tag] = sid

    q_date = "2025-12-01 10:00"
    questions = (
        VirginQuestion(
            qid=f"{spec.conv_id}_q1_needle",
            conversation_id=spec.conv_id,
            domain=spec.domain,
            category="single_session_needle",
            question=spec.needle_question,
            gold_reference=spec.needle_gold,
            question_date=q_date,
            abstention_required=False,
            evidence_turn_ids=(tagged_turn_ids["needle_target"],),
            evidence_session_ids=(tagged_session_ids["needle_target"],),
        ),
        VirginQuestion(
            qid=f"{spec.conv_id}_q2_synthesis",
            conversation_id=spec.conv_id,
            domain=spec.domain,
            category="multi_session_synthesis",
            question=spec.synth_question,
            gold_reference=spec.synth_gold,
            question_date=q_date,
            abstention_required=False,
            evidence_turn_ids=(
                tagged_turn_ids["synth_1"],
                tagged_turn_ids["synth_2"],
                tagged_turn_ids["synth_3"],
            ),
            evidence_session_ids=(
                tagged_session_ids["synth_1"],
                tagged_session_ids["synth_2"],
                tagged_session_ids["synth_3"],
            ),
        ),
        VirginQuestion(
            qid=f"{spec.conv_id}_q3_ku_echo",
            conversation_id=spec.conv_id,
            domain=spec.domain,
            category="knowledge_update_echo_flood",
            question=spec.ku_question,
            gold_reference=spec.ku_gold,
            question_date=q_date,
            abstention_required=False,
            evidence_turn_ids=(tagged_turn_ids["ku_initial"], tagged_turn_ids["ku_update"]),
            evidence_session_ids=(tagged_session_ids["ku_initial"], tagged_session_ids["ku_update"]),
            stale_echo_value=spec.ku_stale_val,
        ),
        VirginQuestion(
            qid=f"{spec.conv_id}_q4_attribution",
            conversation_id=spec.conv_id,
            domain=spec.domain,
            category="multi_speaker_attribution",
            question=spec.attr_question,
            gold_reference=spec.attr_gold,
            question_date=q_date,
            abstention_required=False,
            evidence_turn_ids=(tagged_turn_ids["attr_target"],),
            evidence_session_ids=(tagged_session_ids["attr_target"],),
        ),
        VirginQuestion(
            qid=f"{spec.conv_id}_q5_abstention",
            conversation_id=spec.conv_id,
            domain=spec.domain,
            category="false_premise_abstention",
            question=spec.abs_question,
            gold_reference=spec.abs_gold_note,
            question_date=q_date,
            abstention_required=True,
            evidence_turn_ids=(),
            evidence_session_ids=(),
        ),
    )

    return VirginConversation(
        conversation_id=spec.conv_id,
        domain=spec.domain,
        speakers=(spec.spk_a, spec.spk_b),
        session_dates=tuple(dt for _, dt, _ in session_payloads),
        turns=tuple(turns),
        questions=questions,
    )


def build_virgin_conversational_holdout() -> tuple[VirginConversation, ...]:
    """Build all 24 lineage-disjoint multi-session conversations (`N=120` = `24 x 5` items)."""
    if len(_SCENARIOS) != 24:
        raise RuntimeError(f"Expected 24 Virgin scenarios, found {len(_SCENARIOS)}")
    convs = tuple(_build_conversation(i, spec) for i, spec in enumerate(_SCENARIOS))

    # Structural & disjointness self-verification
    seen_speakers: set[str] = set()
    seen_qids: set[str] = set()
    for conv in convs:
        for sp in conv.speakers:
            cf = sp.casefold()
            if cf in LOCOMO_FORBIDDEN_SPEAKERS:
                raise ValueError(f"Forbidden LoCoMo speaker {sp!r}")
            if cf in seen_speakers:
                raise ValueError(f"Duplicate speaker {sp!r} across conversations")
            seen_speakers.add(cf)
        if not (8 <= len(conv.session_dates) <= 12):
            raise ValueError(f"{conv.conversation_id}: session count {len(conv.session_dates)} outside 8..12")
        if [q.category for q in conv.questions] != list(ITEM_CATEGORIES):
            raise ValueError(f"{conv.conversation_id}: unbalanced question categories")
        for q in conv.questions:
            if q.qid in seen_qids:
                raise ValueError(f"Duplicate qid {q.qid}")
            seen_qids.add(q.qid)

    if len(seen_qids) != 120:
        raise RuntimeError(f"Expected 120 total Virgin items, got {len(seen_qids)}")
    return convs


def virgin_suite_digest(conversations: Sequence[VirginConversation] | None = None) -> str:
    """Deterministic SHA-256 digest over all 24 conversations, turns, dates, and 120 questions."""
    convs = build_virgin_conversational_holdout() if conversations is None else tuple(conversations)
    serializable = [
        {
            "conversation_id": c.conversation_id,
            "domain": c.domain,
            "speakers": list(c.speakers),
            "session_dates": list(c.session_dates),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "session_id": t.session_id,
                    "date": t.date,
                    "timestamp": t.timestamp,
                    "speaker": t.speaker,
                    "text": t.text,
                    "claim_text": t.claim_text,
                }
                for t in c.turns
            ],
            "questions": [
                {
                    "qid": q.qid,
                    "category": q.category,
                    "question": q.question,
                    "gold_reference": q.gold_reference,
                    "question_date": q.question_date,
                    "abstention_required": q.abstention_required,
                    "evidence_turn_ids": list(q.evidence_turn_ids),
                    "evidence_session_ids": list(q.evidence_session_ids),
                }
                for q in c.questions
            ],
        }
        for c in convs
    ]
    blob = json.dumps(serializable, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_lineage_disjointness(
    conversations: Sequence[VirginConversation],
    *,
    locomo_convs: Sequence[dict[str, Any]] | None = None,
    lme_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify zero speaker, question-ID, or normalized question-text overlap with LoCoMo and LME."""
    virgin_speakers = {sp.casefold() for c in conversations for sp in c.speakers}
    virgin_qids = {q.qid for c in conversations for q in c.questions}
    virgin_qtexts = {" ".join(q.question.lower().split()) for c in conversations for q in c.questions}

    loc_speakers: set[str] = set(LOCOMO_FORBIDDEN_SPEAKERS)
    loc_qtexts: set[str] = set()
    if locomo_convs is not None:
        for conv_obj in locomo_convs:
            for val in conv_obj.get("conversation", {}).values():
                if isinstance(val, list):
                    for turn in val:
                        sp = str(turn.get("speaker", "")).strip().casefold()
                        if sp:
                            loc_speakers.add(sp)
            for qa in conv_obj.get("qa", []):
                qt = " ".join(str(qa.get("question", "")).lower().split())
                if qt:
                    loc_qtexts.add(qt)

    lme_qids: set[str] = set()
    lme_qtexts: set[str] = set()
    if lme_items is not None:
        for item in lme_items:
            lme_qids.add(str(item.get("question_id", "")))
            qt = " ".join(str(item.get("question", "")).lower().split())
            if qt:
                lme_qtexts.add(qt)

    speaker_overlap = sorted(virgin_speakers & loc_speakers)
    qid_overlap = sorted(virgin_qids & lme_qids)
    qtext_overlap = sorted(virgin_qtexts & (loc_qtexts | lme_qtexts))

    ok = not speaker_overlap and not qid_overlap and not qtext_overlap
    if not ok:
        raise ValueError(
            f"Lineage disjointness failed: speakers={speaker_overlap}, qids={qid_overlap}, qtexts={qtext_overlap}"
        )
    return {
        "disjoint_verified": True,
        "n_virgin_conversations": len(conversations),
        "n_virgin_questions": sum(len(c.questions) for c in conversations),
        "n_unique_virgin_speakers": len(virgin_speakers),
        "speaker_overlap_with_locomo": speaker_overlap,
        "qid_overlap_with_lme": qid_overlap,
        "question_text_overlap": qtext_overlap,
    }
