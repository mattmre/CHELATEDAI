import os; os.environ["HF_HOME"] = "/home/mattmre/chelated-night1/hf_cache"  # noqa: E702

# Spark 6724 uses the exact R5 answering and judging protocol.

import argparse
import csv
import hashlib
import json
import re
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chelated_conversational_v3 import (  # noqa: E402
    BGE_DIMENSIONS,
    BGE_MODEL,
    ChelatedConversationalV3,
    ConversationTurn,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    write_json_atomic,
)

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
LME, LOCOMO = "LongMemEval-S", "LoCoMo-300"
CLEAN, STRESS, STRESS_R5 = "clean", "echo_stress", "echo_stress_r5"
FLAT = "Legacy_MiniLM_Flat_Top5"
FLAT_BGEM3_TM = "Flat_BGEM3_TokenMatched_Control"
HYBRID = "DualGran_BGEM3_Hybrid_CleanBaseline"
GATED = "Chelated_v3_DualGran_BGEM3_EK5v3"
GATED_J4 = "Chelated_v3_J4_Arch_R1R2R3"
ZERO_CTX = "Retrieve_Nothing_ZeroContext_Control"
ARMS = (FLAT, FLAT_BGEM3_TM, HYBRID, GATED, GATED_J4)
CLEAN_ONLY_CONTROLS = (ZERO_CTX,)

# Pre-registered frozen J4 architectural config (selected strictly from DEV failure taxonomy; never tuned on TEST)
J4_FROZEN_ARCH_CONFIG = {
    "config_id": "pm1b_j4_frozen_r1r2r3_v1",
    "speaker_attribution_header": True,
    "speaker_anchor_reserve": 2,
    "adaptive_rrf_gap": 0.14,
    "min_sessions": 3,
    "max_sessions": 6,
    "session_order": "relevance",
}
J4_SPLIT_SEED = "pm1b_j4_holdout_split_v1"
LME_DATA = ("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json")
LOC_DATA = ("Percena/locomo-mc10", "raw/locomo10.json")
EXPECTED = {
    (LME, CLEAN): 500,
    (LME, STRESS): 78,
    (LME, STRESS_R5): 78,
    (LOCOMO, CLEAN): 300,
    (LOCOMO, STRESS): 300,
}
RETRIEVAL_LME = {"anchor_count": 12, "window_radius": 2, "context_chars": 26000, "pool_size": 36}
RETRIEVAL_LOC = {"anchor_count": 10, "window_radius": 1, "context_chars": 12000, "pool_size": 36}
RETRIEVAL = {"LongMemEval-S": RETRIEVAL_LME, "LoCoMo-300": RETRIEVAL_LOC}
LME_ANSWER = (
    "You are a precise conversational memory assistant. Answer the user's question strictly using the "
    "dated conversation excerpts below.\n"
    "- Verify exact entity and premise match: if the question asks about a specific person, pet, item, or event "
    "that was NEVER mentioned in the excerpts, explicitly state that the user did not mention that information.\n"
    "- When an earlier detail was later updated in a newer dated session (or marked as an earlier superseded record), "
    "report the latest current fact unless the question explicitly asks about the past.\n"
    "- Resolve any relative time expressions against each excerpt's [Session Date: ...] and Current Question Date. "
    "Be concise (1-2 sentences)."
)
LOC_ANSWER = (
    "Answer the question strictly using ONLY the dated conversation snippets below. Be concise (1 short sentence).\n"
    "- Verify exact speaker attribution: NEVER attribute one speaker's statement, possession, hobby, plan, or event "
    "to a different speaker. If the specific person or detail asked about is not supported in the snippets, "
    "or never occurred in the snippets, reply ONLY: UNANSWERABLE.\n"
    "- Convert relative time phrases (e.g. 'yesterday', 'last week', 'last weekend', 'next month', 'last year') "
    "into explicit calendar dates, months, or years using that snippet's [Session Date: ...]."
)
LME_JUDGE = (
    "You are an impartial QA evaluator for LongMemEval. Reply ONLY YES if the Model Answer accurately "
    "provides the core information in the Reference Answer without contradicting it on updated facts. "
    "Reply NO if it gives an outdated/superseded value, contradicts the reference, or fails to answer."
)
LOC_JUDGE = (
    "You are an impartial QA grader. Reply ONLY YES if the Model Answer conveys the same core fact/date/entity "
    "as the Ground-Truth Answer without contradicting it. Reply NO if it is wrong, contradicts, or gives a different value."
)
ABSTENTION_CUES = (
    "unanswerable", "not mentioned", "no information", "doesn't mention", "does not mention",
    "cannot be determined", "don't have information", "never mentioned",
)
ECHO_SYSTEM = (
    "Write six different casual first-person restatements of the marked speaker's factual statements. "
    "Use the surrounding utterances only to resolve pronouns and identify what is being discussed. "
    "Each restatement must preserve all concrete names, quantities, relationships, and facts in the marked "
    "statements. Ignore requests for recommendations and the assistant's advice. Change the phrasing or "
    "sentence structure in each, using a natural conversational tone. State the facts directly, without "
    "headers, speaker labels, timestamps, archival language, or lead-ins about remembering or repeating. "
    "Do not add facts or say that a value has changed. Return only a JSON array of exactly six strings."
)
ECHO_HEADER = re.compile(
    r"^\s*(?:[\d.]+\s|(?:user|assistant|recap|briefing|memory|summary|digest|note|record)[^:]{0,32}:)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Question:
    suite: str
    qid: str
    kind: str
    text: str
    gold: str
    date: str = ""
    group: int | None = None
    abstention: bool = False
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class Job:
    question: Question
    regime: str
    arm: str
    prompt_sha256: str
    selected_ids: tuple[str, ...]
    evidence_any: int | None
    evidence_all: int | None
    gate_active: bool = False
    echo_clusters: tuple[tuple[str, ...], ...] = ()
    supersessions: tuple[tuple[str, str], ...] = ()


class CachedEncoder:
    """Reuse exact text vectors across questions and across clean/stress copies."""

    def __init__(self, model, dimensions: int, batch_size: int):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.vectors = {}

    def encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.vectors))
        if missing:
            encoded = self.model.encode(
                missing, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            if encoded.shape != (len(missing), self.dimensions) or not np.isfinite(encoded).all():
                raise ValueError("Unexpected embedding output shape or nonfinite embedding")
            self.vectors.update(zip(missing, encoded))
        return np.stack([self.vectors[text] for text in texts])

    def release(self):
        self.vectors.clear()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def build_lme_holdout_split_map(items) -> dict[str, str]:
    """Deterministic stratified 50/50 DEV vs HELD_OUT_TEST split across LongMemEval-S question types."""
    by_kind: dict[str, list[str]] = {}
    for item in items:
        qid = str(item["question_id"])
        by_kind.setdefault(str(item["question_type"]), []).append(qid)
    split_map: dict[str, str] = {}
    for _kind, qids in sorted(by_kind.items()):
        ordered = sorted(
            qids,
            key=lambda q: hashlib.sha256(f"{J4_SPLIT_SEED}::{q}".encode("utf-8")).hexdigest(),
        )
        half = len(ordered) // 2
        for q in ordered[:half]:
            split_map[q] = "DEV"
        for q in ordered[half:]:
            split_map[q] = "HELD_OUT_TEST"
    return split_map


def question_split_tag(q: Question, lme_split_map: dict[str, str]) -> str:
    """LoCoMo convs 0..4 -> DEV, convs 5..9 -> HELD_OUT_TEST; LME -> frozen stratified SHA-256 split."""
    if q.suite == LOCOMO:
        return "DEV" if (q.group is not None and int(q.group) < 5) else "HELD_OUT_TEST"
    return lme_split_map[q.qid]


def _flat_bgem3_token_matched_context(engine: ChelatedConversationalV3, question: str, context_chars: int) -> tuple[str, tuple[str, ...], set[str], bool]:
    """R4 Token-Matched Flat Control: turn-only BM25 + BGE-M3 RRF (no session pooling, no window, no EK5, no R1-R3)."""
    if not engine.turns:
        return "", (), set(), False
    q_vec = engine._encode([question])[0]
    turn_scores = engine._turn_bm25.score_query(question)
    from chelated_conversational_v3 import RetrievedTurn, _rrf

    flat_rrf = _rrf(turn_scores, engine._turn_vectors @ q_vec)
    order = [int(i) for i in np.argsort(-flat_rrf, kind="stable")]
    chosen: list[int] = []
    used = 0
    for idx in order:
        rendered = RetrievedTurn(engine.turns[idx]).render()
        cost = len(rendered) + bool(chosen)
        if used + cost > context_chars:
            if len(chosen) >= 3:
                break
            continue
        chosen.append(idx)
        used += cost
    chosen.sort(key=lambda i: (engine.turns[i].timestamp, i))
    ctx = "\n".join(RetrievedTurn(engine.turns[i]).render() for i in chosen)
    ids = tuple(engine.turns[i].turn_id for i in chosen)
    sids = {engine.turns[i].session_id for i in chosen}
    return ctx, ids, sids, True


def lme_time(text: str) -> float:
    cleaned = normalized(re.sub(r"\([^)]*\)", "", text))
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported LongMemEval session date {text!r}")


def lme_memory(item) -> list[ConversationTurn]:
    result = []
    for session_pos, (sid, date, turns) in enumerate(zip(
        item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True,
    )):
        timestamp = lme_time(date)
        i = 0
        while i < len(turns):
            position = i
            role = turns[i]["role"].capitalize()
            body = turns[i]["content"].strip()
            text = f"{role}: {body}"
            index_text = text
            i += 1
            if role == "User" and i < len(turns) and turns[i]["role"] == "assistant":
                reply = turns[i]["content"].strip()
                text += f" | Assistant: {reply}"
                index_text += f" | Assistant: {reply[:360]}"
                i += 1
            result.append(ConversationTurn(
                f"{sid}:{session_pos}:{position}", sid, date, timestamp + position * 0.001,
                text, body, role, index_text,
            ))
    return result


def _echo_source(item):
    # The existing annotated evidence identifies the oldest witness for this
    # controlled perturbation. The generator receives neither the question nor gold answer.
    answer_ids = set(item["answer_session_ids"])
    dated = sorted(
        (lme_time(date), pos, sid, turns)
        for pos, (sid, date, turns) in enumerate(zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True,
        ))
        if sid in answer_ids
    )
    earlier, _, sid, turns = dated[0]
    latest = dated[-1][0]
    if latest <= earlier:
        raise ValueError(f"Stress item {item['question_id']} has no strict source chronology")
    marked = [t["content"] for t in turns if t.get("has_answer") and t["role"] == "user"]
    # Some annotated sessions have no span marks, including abstention cases.
    # Use only an actual user utterance from the earlier session in that case.
    provenance = "annotated_user_span" if marked else "unmarked_earlier_session_user_turn"
    if not marked:
        user_turns = [t["content"] for t in turns if t["role"] == "user"]
        marked = [max(user_turns, key=len)]
    dialogue = "\n".join(
        f"{'MARKED ' if t['role'] == 'user' and t['content'] in marked else ''}{t['role']}: {t['content']}"
        for t in turns
    )
    return earlier, latest, sid, dialogue, marked, provenance


def echo_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.lower()))


def parse_echoes(text: str, source: list[str]) -> tuple[str, ...]:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("[")
    if start < 0:
        raise ValueError("Paraphrase response did not include a JSON array")
    values, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(values, list) or len(values) != 6 or any(not isinstance(v, str) for v in values):
        raise ValueError("Expected six paraphrase strings")
    echoes = tuple(normalized(v) for v in values)
    echo_keys = {echo_words(v) for v in echoes}
    source_keys = {echo_words(v) for v in source}
    if len(echo_keys) != 6 or echo_keys & source_keys or any(
        len(v.split()) < 4 or ECHO_HEADER.match(v) for v in echoes
    ):
        raise ValueError("Expected distinct, header-free restatements")
    return echoes


def make_lme_echoes(item, call):
    earlier, latest, source_sid, dialogue, marked, provenance = _echo_source(item)
    prompt = f"Conversation:\n{dialogue}\n\nReturn the six casual paraphrases as JSON."
    for attempt in range(2):
        text, _, _, error = call(ECHO_SYSTEM, prompt, 1400)
        try:
            if error:
                raise ValueError(text)
            echoes = parse_echoes(text, marked)
            break
        except ValueError:
            prompt = (
                f"Conversation:\n{dialogue}\n\nReturn a compact JSON array with six distinct sentences. "
                "Each must directly preserve the same source facts using substantially different wording. "
                "Begin with [ and end with ]. No speaker labels or formatting."
            )
    else:
        # Some completions cannot reliably emit a whole array. Collect one
        # independent, differently worded direct restatement per request.
        instructions = (
            "Rewrite the supplied speaker text as one casual first-person utterance. "
            "Keep its subject, facts, names, and quantities. Rephrase requests as the speaker's own "
            "interests or goals. Avoid advice, added details, quotation marks, and labels. "
            "Return only the rewritten utterance."
        )
        source_text = "\n".join(marked)
        styles = (
            "Start with the speaker's current activity", "Lead with the main subject",
            "Start with the speaker's goal", "Use a short conversational sentence",
            "Use two natural sentences", "Put the supporting detail first",
            "Open with a different verb", "Change the order of the ideas",
            "Supply the brief version", "Use a more relaxed conversational tone",
        )
        alternatives = []
        seen = {echo_words(text) for text in marked}
        for style in styles:
            prior = "\n".join(f"- {text}" for text in alternatives)
            prompt = (
                f"Speaker text:\n{source_text}\n\nPhrasings already used:\n{prior}\n\n{style}. "
                "Stay faithful to the speaker text."
            )
            text, _, _, error = call(instructions, prompt, 400)
            text = normalized(text.strip().strip('"'))
            words = echo_words(text)
            if not error and len(text.split()) >= 4 and words not in seen and not ECHO_HEADER.match(text):
                seen.add(words)
                alternatives.append(text)
            if len(alternatives) == 6:
                echoes = parse_echoes(json.dumps(alternatives), marked)
                break
        else:
            raise ValueError(f"{item['question_id']}: failed to collect six valid natural paraphrases")
    turns = []
    for i, echo in enumerate(echoes):
        timestamp = earlier + (latest - earlier) * (i + 1) / 7
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y/%m/%d (%a) %H:%M")
        turns.append(ConversationTurn(
            f"echo_{i}", f"echo_session_{i}", date, timestamp, f"User: {echo}", echo, "User",
        ))
    return {"qid": str(item["question_id"]), "source_session_id": source_sid,
            "source_provenance": provenance, "turns": [asdict(t) for t in turns]}


def r5_locomo_questions(conversations):
    questions = []
    for ci, conv in enumerate(conversations):
        strata = [[], [], []]
        for qi, qa in enumerate(conv["qa"]):
            cat = int(qa["category"])
            qid = f"loc_{ci}_{qi}"
            evidence = tuple(dict.fromkeys(
                tok for item in qa.get("evidence", []) for tok in re.split(r"[;,\s]+", str(item).strip()) if tok
            ))
            q = Question(
                LOCOMO, qid, str(cat), str(qa["question"]),
                "UNANSWERABLE" if cat == 5 else str(qa.get("answer", "")),
                group=ci, abstention=cat == 5, evidence=evidence,
            )
            if cat == 2:
                strata[0].append(q)
            elif cat in (1, 4):
                strata[1].append(q)
            elif cat == 5:
                strata[2].append(q)
        for bucket in strata:
            if len(bucket) < 10:
                raise ValueError(f"LoCoMo conversation {ci} cannot supply the R5 balanced sample")
            questions.extend(sorted(bucket, key=lambda q: hashlib.sha256(q.qid.encode()).hexdigest())[:10])
    return questions


def r5_legacy_row(turn: ConversationTurn) -> dict:
    return {
        "unit_id": turn.turn_id, "session_id": turn.session_id, "date_str": turn.date,
        "ts": turn.timestamp, "text": turn.text, "raw_text": turn.claim_text, "user_text": turn.claim_text,
    }


def split_legacy_row(row) -> ConversationTurn:
    return ConversationTurn(
        row["unit_id"], row["session_id"], row["date_str"], row["ts"], row["text"], row["raw_text"],
        row["text"].partition(":")[0],
    )


def neutral_locomo_echoes(turns, vectors):
    centroid = vectors.mean(axis=0)
    similarity = vectors @ (centroid / max(float(np.linalg.norm(centroid)), 1e-12))
    candidates = [int(i) for i in np.argsort(-similarity) if len(turns[i].claim_text.split()) >= 12][:15]
    if len(candidates) != 15:
        raise ValueError("LoCoMo neutral stress needs 15 question-blind source turns")
    return [
        replace(
            turns[index],
            turn_id=f"echo_{index}_{rep}",
            session_id=f"echo_session_{index}_{rep}",
            timestamp=turns[index].timestamp + 0.01 * (rep + 1),
        )
        for index in candidates for rep in range(6)
    ]


def answer_prompt(q: Question, context: str) -> str:
    if q.suite == LME:
        return f"Current Question Date: {q.date}\nConversation Excerpts:\n{context}\n\nQuestion: {q.text}\nAnswer:"
    return f"Conversation Snippets:\n{context}\n\nQuestion: {q.text}\nAnswer:"


def r5_grade(q: Question, prediction: str):
    """Return an R5 shortcut grade or the exact R5 judge system/user messages."""
    lower = prediction.lower()
    if not lower or lower.startswith("error:"):
        return False
    if q.suite == LME:
        gold = normalized(q.gold).lower()
        if len(gold) >= 4 and gold in lower and len(lower) <= len(gold) + 70:
            return True
        return (
            LME_JUDGE,
            f"Question: {q.text}\nReference Answer: {q.gold}\nModel Answer: {prediction}\n\nCorrect? Reply YES or NO.",
        )
    refusal = any(cue in lower for cue in ABSTENTION_CUES)
    if q.abstention or (refusal and len(lower) < 90):
        return q.abstention and refusal
    return (
        LOC_JUDGE,
        f"Question: {q.text}\nGround-Truth Answer: {q.gold}\nModel Answer: {prediction}\n\nMatch? Reply YES or NO.",
    )


class Benchmark:
    def __init__(self, flat, bge, call, workers, output):
        self.flat = flat
        self.bge = bge
        self.call = call
        self.workers = workers
        self.output = output
        self.prompts = {}
        self.jobs = []

    def add(self, q, regime, arm, context, ids, evidence, recall=None):
        system, tokens = (LME_ANSWER, 96) if q.suite == LME else (LOC_ANSWER, 64)
        request = (system, answer_prompt(q, context), tokens)
        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode()).hexdigest()
        self.prompts.setdefault(digest, request)
        labeled = set(q.evidence) if not q.abstention else set()
        self.jobs.append(Job(
            q, regime, arm, digest, tuple(ids),
            int(bool(labeled.intersection(evidence))) if labeled else None,
            int(labeled.issubset(evidence)) if labeled else None,
            recall.gate_active if recall else False,
            recall.echo_clusters if recall else (),
            recall.supersessions if recall else (),
        ))

    def rank_regime(self, questions, regime, legacy, turns):
        from scripts.run_overnight_r5_spark6724 import TurnBM25, _retrieve_lme_r5_arm

        text_bank = [row["text"] for row in legacy]
        bm25 = TurnBM25(text_bank)
        flat_vectors = self.flat.encode(text_bank)
        flat_queries = self.flat.encode([q.text for q in questions])
        engine = ChelatedConversationalV3(turns, self.bge)
        for q, vector in zip(questions, flat_queries, strict=True):
            indices, _ = _retrieve_lme_r5_arm("M1_hybrid_flat", legacy, flat_vectors, vector, bm25.score(q.text))
            ordered = sorted(indices, key=lambda i: legacy[i]["ts"])
            rows = [legacy[i] for i in ordered]
            flat_ctx = "\n".join(f"- [{row['date_str']}] {row['text']}" for row in rows)
            evidence_key = "session_id" if q.suite == LME else "unit_id"
            self.add(q, regime, FLAT, flat_ctx, (row["unit_id"] for row in rows),
                     {row[evidence_key] for row in rows})
            ret_cfg = RETRIEVAL[q.suite]
            tm_ctx, tm_ids, tm_sids, _ = _flat_bgem3_token_matched_context(engine, q.text, ret_cfg["context_chars"])
            tm_ev = tm_sids if q.suite == LME else set(tm_ids)
            self.add(q, regime, FLAT_BGEM3_TM, tm_ctx, tm_ids, tm_ev)
            if regime == CLEAN:
                self.add(q, regime, ZERO_CTX, "(No conversation excerpts retrieved.)", (), set())
            ranking, _, sess_rrf = engine.rank_with_scores(q.text)
            clean = engine.retrieve(q.text, epistemic=False, ranked=ranking, **ret_cfg)
            gated = engine.retrieve(q.text, ranked=ranking, **ret_cfg)
            j4_kwargs = {k: v for k, v in J4_FROZEN_ARCH_CONFIG.items() if k != "config_id"}
            if q.suite == LME:
                # LME speakers are User/Assistant (not named dialogue cast), so R1 header is naturally a no-op
                j4_kwargs["speaker_attribution_header"] = False
                j4_kwargs["speaker_anchor_reserve"] = 0
            gated_j4 = engine.retrieve(
                q.text,
                ranked=ranking,
                session_rrf=sess_rrf,
                **ret_cfg,
                **j4_kwargs,
            )
            if not gated.gate_active and clean != gated:
                raise RuntimeError(f"Inactive gate changed the hybrid result for {q.qid}")
            for arm, recall in ((HYBRID, clean), (GATED, gated), (GATED_J4, gated_j4)):
                ids = [entry.turn.turn_id for entry in recall.turns]
                evidence = {entry.turn.session_id for entry in recall.turns} if q.suite == LME else set(ids)
                self.add(q, regime, arm, recall.context, ids, evidence, recall)

    def lme(self, items, echo_by_id):
        from scripts.run_overnight_r5_spark6724 import _extract_lme_units

        for i, item in enumerate(items, 1):
            q = Question(
                LME, str(item["question_id"]), item["question_type"], item["question"], str(item["answer"]),
                date=item["question_date"], abstention=str(item["question_id"]).endswith("_abs"),
                evidence=tuple(item["answer_session_ids"]),
            )
            legacy, turns = _extract_lme_units(item), lme_memory(item)
            self.rank_regime([q], CLEAN, legacy, turns)
            if q.qid in echo_by_id:
                echoes = [ConversationTurn(**t) for t in echo_by_id[q.qid]["turns"]]
                stressed = sorted(legacy + [r5_legacy_row(t) for t in echoes], key=lambda r: (r["ts"], r["unit_id"]))
                self.rank_regime([q], STRESS, stressed, turns + echoes)
                legacy_r5 = _extract_lme_units(item, inject_echo_stress=True)
                echo_r5 = [
                    ConversationTurn(
                        u["unit_id"], u["session_id"], u["date_str"], u["ts"], u["text"], u["raw_text"], "User",
                    )
                    for u in legacy_r5 if str(u["unit_id"]).startswith("echo_")
                ]
                self.rank_regime([q], STRESS_R5, legacy_r5, turns + echo_r5)
            if i % 25 == 0:
                print(f"LongMemEval-S indexed {i}/{len(items)} items, {len(self.prompts)} answer prompts", flush=True)

    def locomo(self, conversations):
        from scripts.run_flagship2_r4_ek5_v3_and_suites import _extract_locomo_units

        questions = r5_locomo_questions(conversations)
        for i, conv in enumerate(conversations):
            legacy = _extract_locomo_units(conv)
            turns = [split_legacy_row(row) for row in legacy]
            subset = [q for q in questions if q.group == i]
            vectors = self.flat.encode([row["text"] for row in legacy])
            echoes = neutral_locomo_echoes(turns, vectors)
            self.rank_regime(subset, CLEAN, legacy, turns)
            self.rank_regime(subset, STRESS, legacy + [r5_legacy_row(t) for t in echoes], turns + echoes)
            print(f"LoCoMo indexed conversation {i + 1}/{len(conversations)}", flush=True)

    def finish(self, lme_split_map):
        # The same prompt and normalized (dataset, qid, prediction) share model outputs across arms.
        checkpoint = self.output / "r6_pm1b_answer_cache.json"
        answers = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
        todo = [digest for digest in self.prompts if digest not in answers or answers[digest][3]]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i in range(0, len(todo), 100):
                keys = todo[i:i + 100]
                values = pool.map(lambda digest: self.call(*self.prompts[digest]), keys)
                answers.update(zip(keys, values, strict=True))
                write_json_atomic(checkpoint, answers)
                print(f"Completed {min(i + 100, len(todo))}/{len(todo)} pending answer prompts", flush=True)

            grade_cache_path = self.output / "r6_pm1b_grade_cache.json"
            raw_grade_cache = json.loads(grade_cache_path.read_text(encoding="utf-8")) if grade_cache_path.exists() else {}
            grades = {}
            for k_str, val in raw_grade_cache.items():
                parts = k_str.split("|||", 2)
                if len(parts) == 3 and not val[1]:
                    grades[(parts[0], parts[1], parts[2])] = (bool(val[0]), False)

            judge_jobs = {}
            shortcut_hits = 0
            for job in self.jobs:
                q = job.question
                pred = normalized(answers[job.prompt_sha256][0])
                key = (q.suite, q.qid, pred.lower())
                if key not in grades and key not in judge_jobs:
                    grade = r5_grade(q, pred)
                    if isinstance(grade, bool):
                        grades[key] = (grade, False)
                        shortcut_hits += 1
                    else:
                        judge_jobs[key] = grade
            print(f"Grading {len(judge_jobs)} uncached unique R5 judge prompts ({shortcut_hits} deterministic shortcut grades)", flush=True)
            if judge_jobs:
                responses = pool.map(lambda args: self.call(*args, 6), judge_jobs.values())
                for key, result in zip(judge_jobs, responses, strict=True):
                    grades[key] = (result[0].strip().upper().startswith("YES"), result[3])
                serial_grades = {f"{k[0]}|||{k[1]}|||{k[2]}": [bool(v[0]), bool(v[1])] for k, v in grades.items()}
                write_json_atomic(grade_cache_path, serial_grades)

        rows = []
        for job in self.jobs:
            q = job.question
            pred, tokens, latency, error = answers[job.prompt_sha256]
            pred = normalized(pred)
            correct, judge_error = grades[q.suite, q.qid, pred.lower()]
            rows.append({
                "suite": q.suite, "regime": job.regime, "split": question_split_tag(q, lme_split_map),
                "arm": job.arm, "qid": q.qid,
                "question_type": q.kind, "conversation": q.group, "is_abstention": int(q.abstention),
                "gold": normalized(q.gold), "prediction": pred, "is_correct": int(correct),
                "evidence_any": job.evidence_any, "evidence_all": job.evidence_all,
                "prompt_tokens": tokens, "latency_ms": round(latency, 2),
                "api_error": int(error or judge_error), "prompt_sha256": job.prompt_sha256,
                "gate_active": int(job.gate_active), "selected_ids": json.dumps(job.selected_ids),
                "echo_clusters": json.dumps(job.echo_clusters), "supersessions": json.dumps(job.supersessions),
            })
        return rows


def paired(candidate, reference, ids):
    def outcomes(rows):
        result = {r["qid"]: r["is_correct"] for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate qid within a benchmark arm")
        return result

    return paired_bootstrap_delta(
        outcomes(candidate), outcomes(reference), n_bootstrap=3000, seed=42, expected_ids=ids,
    )


def rate(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return sum(values) / len(values) if values else None


def _summarize_subset(rows, expected_ids_by_pair, *, include_clean_zero_ctx: bool = True):
    from scripts.run_flagship2_r4_ek5_v3_and_suites import _cluster_bootstrap_delta

    suites = {}
    pair_list = (
        (GATED, FLAT),
        (GATED, HYBRID),
        (HYBRID, FLAT),
        (GATED_J4, FLAT),
        (GATED_J4, FLAT_BGEM3_TM),
        (GATED_J4, GATED),
        (GATED_J4, HYBRID),
        (FLAT_BGEM3_TM, FLAT),
    )
    for (suite, regime), ids in expected_ids_by_pair.items():
        if not ids:
            continue
        active_arms = ARMS + (CLEAN_ONLY_CONTROLS if (include_clean_zero_ctx and regime == CLEAN) else ())
        by_arm = {
            arm: [r for r in rows if (r["suite"], r["regime"], r["arm"]) == (suite, regime, arm)]
            for arm in active_arms
        }
        if any(len(arm_rows) != len(ids) for arm_rows in by_arm.values()):
            raise ValueError(f"{suite}/{regime}: incomplete arm coverage ({ {k: len(v) for k, v in by_arm.items()} } vs {len(ids)})")
        comparisons = {}
        for a, b in pair_list:
            comparison = paired(by_arm[a], by_arm[b], ids)
            if suite == LOCOMO:
                cluster = _cluster_bootstrap_delta(
                    {r["qid"]: r["is_correct"] for r in by_arm[a]},
                    {r["qid"]: r["is_correct"] for r in by_arm[b]},
                    {r["qid"]: r["conversation"] for r in by_arm[a]}, n_bootstrap=3000, seed=42,
                )
                comparison.update({key: value for key, value in cluster.items() if key.startswith("cluster_")})
            comparisons[f"{a}_vs_{b}"] = comparison
        hybrid_by_id = {r["qid"]: r for r in by_arm[HYBRID]}
        gated_rows = by_arm[GATED]
        inactive = [r for r in gated_rows if not r["gate_active"]]
        if any(r["prompt_sha256"] != hybrid_by_id[r["qid"]]["prompt_sha256"] for r in inactive):
            raise ValueError("Inactive-gate prompt parity failed")
        types = sorted({r["question_type"] for r in gated_rows})
        suites.setdefault(suite, {})[regime] = {
            "n": len(ids), "comparisons": comparisons,
            "same_hybrid_prompt_n": sum(
                r["prompt_sha256"] == hybrid_by_id[r["qid"]]["prompt_sha256"] for r in gated_rows
            ),
            "inactive_gate_prompt_parity_n": len(inactive),
            "arms": {
                a: {
                    "n_correct": sum(r["is_correct"] for r in arm_rows),
                    "accuracy": rate(arm_rows, "is_correct"),
                    "evidence_any_recall": rate(arm_rows, "evidence_any"),
                    "evidence_all_recall": rate(arm_rows, "evidence_all"),
                    "evidence_n": sum(r["evidence_any"] is not None for r in arm_rows),
                    "mean_prompt_tokens": rate(arm_rows, "prompt_tokens"),
                    "api_errors": sum(r["api_error"] for r in arm_rows),
                    "gate_active": sum(r["gate_active"] for r in arm_rows),
                    "by_question_type": {
                        kind: rate([r for r in arm_rows if r["question_type"] == kind], "is_correct")
                        for kind in types
                    },
                } for a, arm_rows in by_arm.items()
            },
        }
    return suites


def summarize(rows, expected_ids):
    suites = _summarize_subset(rows, expected_ids, include_clean_zero_ctx=True)

    # Partition IDs into PRIMARY CONFIRMATORY (`HELD_OUT_TEST`: LoCoMo convs 5..9 & LME test half) vs `DEV`
    split_by_sqid = {(r["suite"], r["qid"]): r["split"] for r in rows}
    heldout_ids = {
        k: [qid for qid in ids if split_by_sqid.get((k[0], qid)) == "HELD_OUT_TEST"]
        for k, ids in expected_ids.items()
    }
    dev_ids = {
        k: [qid for qid in ids if split_by_sqid.get((k[0], qid)) == "DEV"]
        for k, ids in expected_ids.items()
    }
    heldout_suites = _summarize_subset(
        [r for r in rows if r["split"] == "HELD_OUT_TEST"], heldout_ids, include_clean_zero_ctx=True
    )
    dev_suites = _summarize_subset(
        [r for r in rows if r["split"] == "DEV"], dev_ids, include_clean_zero_ctx=True
    )

    ssu = [
        r for r in rows if r["suite"] == LME and r["regime"] == CLEAN and r["question_type"] == "single-session-user"
    ]
    ssu_ids = [r["qid"] for r in ssu if r["arm"] == FLAT]
    if len(ssu_ids) != 70:
        raise ValueError("Expected the R5 single-session-user slice of 70 items")
    ssu_delta = paired([r for r in ssu if r["arm"] == GATED], [r for r in ssu if r["arm"] == FLAT], ssu_ids)
    ssu_j4_delta = paired([r for r in ssu if r["arm"] == GATED_J4], [r for r in ssu if r["arm"] == FLAT], ssu_ids)
    ssu_ho = [r for r in ssu if r["split"] == "HELD_OUT_TEST"]
    ssu_ho_ids = [r["qid"] for r in ssu_ho if r["arm"] == FLAT]
    ssu_ho_j4_delta = paired([r for r in ssu_ho if r["arm"] == GATED_J4], [r for r in ssu_ho if r["arm"] == FLAT], ssu_ho_ids)

    expected_row_total = sum(EXPECTED.values()) * len(ARMS) + (EXPECTED[(LME, CLEAN)] + EXPECTED[(LOCOMO, CLEAN)]) * len(CLEAN_ONLY_CONTROLS)
    flat_comparison, hybrid_comparison = f"{GATED}_vs_{FLAT}", f"{GATED}_vs_{HYBRID}"
    j4_flat_comp, j4_hyb_comp = f"{GATED_J4}_vs_{FLAT}", f"{GATED_J4}_vs_{HYBRID}"
    lme_clean = suites[LME][CLEAN]["comparisons"]
    lme_stress_nat = suites[LME][STRESS]["comparisons"][flat_comparison]
    lme_stress_r5 = suites[LME][STRESS_R5]["comparisons"][flat_comparison]
    loc_clean = suites[LOCOMO][CLEAN]["comparisons"]
    loc_stress = suites[LOCOMO][STRESS]["comparisons"][flat_comparison]

    checks = {
        "coverage_and_zero_api_errors": len(rows) == expected_row_total and not any(r["api_error"] for r in rows),
        "lme_clean_vs_m1_mean_ge_minus_0_01": lme_clean[flat_comparison]["mean_delta"] >= -0.01,
        "lme_clean_vs_m1_lcb95_ge_minus_0_03": lme_clean[flat_comparison]["lcb_95"] >= -0.03,
        "lme_ssu70_vs_m1_mean_ge_minus_0_01": ssu_delta["mean_delta"] >= -0.01,
        "lme_clean_vs_hybrid_mean_ge_zero": lme_clean[hybrid_comparison]["mean_delta"] >= 0,
        "lme_ku_stress_r5_vs_m1_mean_ge_0_25": lme_stress_r5["mean_delta"] >= 0.25,
        "lme_ku_stress_r5_vs_m1_lcb95_ge_0_15": lme_stress_r5["lcb_95"] >= 0.15,
        "lme_ku_stress_nat_vs_m1_mean_gt_zero": lme_stress_nat["mean_delta"] > 0,
        "locomo_clean_vs_m1_mean_ge_minus_0_01": loc_clean[flat_comparison]["mean_delta"] >= -0.01,
        "locomo_clean_vs_hybrid_mean_ge_zero": loc_clean[hybrid_comparison]["mean_delta"] >= 0,
        "locomo_stress_vs_m1_mean_gt_zero": loc_stress["mean_delta"] > 0,
    }
    j4_heldout_checks = {
        "lme_heldout_clean_j4_vs_m1_mean_ge_minus_0_01": heldout_suites[LME][CLEAN]["comparisons"][j4_flat_comp]["mean_delta"] >= -0.01,
        "lme_heldout_ssu35_j4_vs_m1_mean_ge_minus_0_01": ssu_ho_j4_delta["mean_delta"] >= -0.01,
        "lme_heldout_ku_r5_j4_vs_m1_mean_ge_0_25": heldout_suites[LME][STRESS_R5]["comparisons"][j4_flat_comp]["mean_delta"] >= 0.25,
        "locomo_heldout_clean_j4_vs_m1_mean_ge_minus_0_01": heldout_suites[LOCOMO][CLEAN]["comparisons"][j4_flat_comp]["mean_delta"] >= -0.01,
        "locomo_heldout_stress_j4_vs_m1_mean_gt_zero": heldout_suites[LOCOMO][STRESS]["comparisons"][j4_flat_comp]["mean_delta"] > 0,
        "j4_clean_lme_ek5_delta_is_dualgran_not_ek5_disclosed": True,
    }
    return {
        "primary_held_out_confirmatory": {
            "split_definition": "LoCoMo conversations 5..9 (N=150) + LongMemEval-S frozen SHA-256 test half (N=250, KU=39, SSU=35)",
            "frozen_j4_config": J4_FROZEN_ARCH_CONFIG,
            "suites": heldout_suites,
            "single_session_user_j4_vs_m1_heldout_n35": ssu_ho_j4_delta,
            "j4_heldout_checks": j4_heldout_checks,
        },
        "dev_tuning_split": {
            "split_definition": "LoCoMo conversations 0..4 (N=150) + LongMemEval-S frozen SHA-256 dev half (N=250)",
            "suites": dev_suites,
        },
        "secondary_pooled_full_dataset_label": "SECONDARY_POOLED_CONTAINS_DEV_DO_NOT_CITE_AS_HELD_OUT",
        "suites": suites,
        "single_session_user_vs_m1_n70": ssu_delta,
        "single_session_user_j4_vs_m1_n70": ssu_j4_delta,
        "checks": checks,
        "assignment_gates_pass": all(checks.values()),
        "j4_attribution_and_shortcut_disclosure": {
            "clean_lme_gain_source": "Clean LongMemEval-S gain (+7.80 pp in v3) comes entirely from DualGran BGE-M3 + session windowing (clean EK5 gate delta = 0.00 pp).",
            "r5_grade_shortcut_note": "r5_grade uses exact R5 deterministic substring/refusal shortcuts before calling the LLM judge; both answerer and automated judge use Nemotron-30B on Spark #2.",
        },
        "additional_ci_checks": {
            "locomo_clean_vs_hybrid_tango_lcb95_ge_zero": loc_clean[hybrid_comparison]["tango_score_lcb95"] >= 0,
            "locomo_stress_vs_m1_lcb95_gt_0_10": loc_stress["lcb_95"] > 0.10,
            "lme_ku_stress_r5_vs_m1_tango_lcb95_ge_0_15": lme_stress_r5["tango_score_lcb95"] >= 0.15,
            "lme_ku_stress_nat_vs_m1_mean_delta": lme_stress_nat["mean_delta"],
            "j4_pooled_locomo_clean_vs_hybrid_mean_delta": suites[LOCOMO][CLEAN]["comparisons"][j4_hyb_comp]["mean_delta"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run the R6 PM-1b benchmark on Spark 6724")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "r6_pm1b_conversational_sota")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890/v1")
    parser.add_argument("--model", default="nemotron-lightning")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-echo-fixture", type=Path, help="Import validated rows from a prior partial run")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("worker and embedding batch counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from sentence_transformers import SentenceTransformer

    from scripts.run_overnight_r5_spark6724 import _call_vllm

    start = time.perf_counter()
    dataset_files = {
        name: Path(hf_hub_download(repo, name_on_hub, repo_type="dataset", local_files_only=True))
        for name, (repo, name_on_hub) in ((LME, LME_DATA), (LOCOMO, LOC_DATA))
    }
    data_hashes = {name: sha256(path) for name, path in dataset_files.items()}
    lme_items, loc_convs = [json.loads(dataset_files[name].read_text(encoding="utf-8")) for name in (LME, LOCOMO)]
    ku_items = [item for item in lme_items if item["question_type"] == "knowledge-update"]
    loc_qids = [q.qid for q in r5_locomo_questions(loc_convs)]
    expected_ids = {
        (LME, CLEAN): [str(item["question_id"]) for item in lme_items],
        (LME, STRESS): [str(item["question_id"]) for item in ku_items],
        (LME, STRESS_R5): [str(item["question_id"]) for item in ku_items],
        (LOCOMO, CLEAN): loc_qids, (LOCOMO, STRESS): loc_qids,
    }
    if len(loc_convs) != 10 or any(len(set(ids)) != EXPECTED[key] for key, ids in expected_ids.items()):
        raise ValueError("Expected all 500 LME, all 78 KU, and 300 balanced questions across 10 LoCoMo conversations")

    def call(system, prompt, tokens):
        return _call_vllm(args.endpoint, args.model, system, prompt, max_tokens=tokens)

    status, _, _, err = call("Reply with OK.", "Reply with OK.", 8)
    if err or not status:
        raise RuntimeError(f"Spark vLLM preflight failed: {status}")

    echo_path = args.output / "r6_pm1b_ku_echo_fixture.json"
    echo_meta = {
        "dataset_sha256": data_hashes[LME], "model": args.model,
        "source_and_generation_code_sha256": sha256(Path(__file__)),
    }
    if args.resume_echo_fixture:
        echo_meta["inherited_fixture_sha256"] = sha256(args.resume_echo_fixture)
    if echo_path.exists():
        fixture = json.loads(echo_path.read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "model"):
            if fixture["provenance"].get(key) != echo_meta[key]:
                raise ValueError(f"KU fixture {key} differs. Use a fresh output directory.")
    else:
        fixture = {"provenance": echo_meta, "items": []}
        if args.resume_echo_fixture:
            prior = json.loads(args.resume_echo_fixture.read_text(encoding="utf-8"))
            for key in ("dataset_sha256", "model"):
                if prior["provenance"][key] != echo_meta[key]:
                    raise ValueError(f"Prior KU fixture has different {key}")
            by_qid = {str(item["question_id"]): item for item in ku_items}
            for record in prior["items"]:
                earlier, latest, sid, _, marked, provenance = _echo_source(by_qid[record["qid"]])
                turns = record["turns"]
                parse_echoes(json.dumps([t["claim_text"] for t in turns]), marked)
                if (sid, provenance) != (record["source_session_id"], record["source_provenance"]):
                    raise ValueError(f"{record['qid']}: prior fixture source changed")
                for i, turn in enumerate(turns):
                    timestamp = earlier + (latest - earlier) * (i + 1) / 7
                    if abs(turn["timestamp"] - timestamp) > 1e-5:
                        raise ValueError(f"{record['qid']}: prior fixture timestamps changed")
                fixture["items"].append(record)
            write_json_atomic(echo_path, fixture)
    echo_by_id = {r["qid"]: r for r in fixture["items"]}
    if len(echo_by_id) != len(fixture["items"]):
        raise ValueError("Duplicate KU fixture IDs")
    missing = [item for item in ku_items if str(item["question_id"]) not in echo_by_id]
    if missing:
        print(f"Generating six natural paraphrases each for {len(missing)} KU source sessions", flush=True)
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(make_lme_echoes, item, call) for item in missing]
            for future in as_completed(futures):
                try:
                    echo = future.result()
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                echo_by_id[echo["qid"]] = echo
                fixture["items"].append(echo)
                fixture["items"].sort(key=lambda record: record["qid"])
                write_json_atomic(echo_path, fixture)
        if errors:
            raise ValueError("\n".join(errors))
    if set(echo_by_id) != set(expected_ids[LME, STRESS]) or any(
        len(record["turns"]) != 6 for record in echo_by_id.values()
    ):
        raise ValueError("Incomplete or unexpected KU echo fixture")
    print(f"Frozen KU fixture: {len(echo_by_id)} items, {6 * len(echo_by_id)} echoes", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("The requested Spark CUDA device is unavailable")
    model_snapshots = {
        model: Path(snapshot_download(model, local_files_only=True)) for model in (BGE_MODEL, MINILM)
    }
    flat_model = SentenceTransformer(str(model_snapshots[MINILM]), device="cuda", local_files_only=True)
    bge_model = SentenceTransformer(
        str(model_snapshots[BGE_MODEL]), device="cuda", local_files_only=True,
        model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
    )
    bge_model.max_seq_length = 384
    print(f"Loaded embedding models on {torch.cuda.get_device_name(0)}", flush=True)
    benchmark = Benchmark(
        CachedEncoder(flat_model, 384, args.batch_size), CachedEncoder(bge_model, BGE_DIMENSIONS, args.batch_size),
        call, args.workers, args.output,
    )
    lme_split_map = build_lme_holdout_split_map(lme_items)
    split_manifest_sha = hashlib.sha256(
        json.dumps({"seed": J4_SPLIT_SEED, "lme_split": lme_split_map, "config": J4_FROZEN_ARCH_CONFIG}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    benchmark.locomo(loc_convs)
    benchmark.lme(lme_items, echo_by_id)
    benchmark.flat.release()
    benchmark.bge.release()
    print(f"Indexed all items, {len(benchmark.prompts)} unique answer prompts for {len(benchmark.jobs)} rows", flush=True)
    rows = benchmark.finish(lme_split_map)
    results = summarize(rows, expected_ids)

    csv_path = args.output / "r6_pm1b_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import chelated_conversational_v3
    import ek5_v3_stratified
    import em4_soft_telemetry_routing
    from scripts import night1_common, run_flagship2_r4_ek5_v3_and_suites, run_overnight_r5_spark6724

    source_files = [Path(__file__)] + [Path(module.__file__) for module in (
        chelated_conversational_v3, ek5_v3_stratified, em4_soft_telemetry_routing,
        night1_common, run_flagship2_r4_ek5_v3_and_suites, run_overnight_r5_spark6724,
    )]
    manifest = {
        "schema_version": "r6_pm1b_conversational_sota_v2_j4_holdout",
        "git_base_sha": current_git_sha(ROOT), "host": socket.gethostname(), "model": args.model,
        "frozen_j4_split_and_config_sha256": split_manifest_sha,
        "wall_seconds": round(time.perf_counter() - start, 2), "n_rows": len(rows),
        "source_sha256": {path.name: sha256(path) for path in source_files},
        "datasets": {
            name: {"sha256": digest, "filename": dataset_files[name].name}
            for name, digest in data_hashes.items()
        },
        "encoders": {
            BGE_MODEL: {"dimensions": BGE_DIMENSIONS, "revision": model_snapshots[BGE_MODEL].name,
                        "dtype": "float16", "max_seq_length": 384, "session_pooling": "mean normalized turn vectors"},
            MINILM: {"dimensions": 384, "revision": model_snapshots[MINILM].name, "dtype": "float32"},
        },
        "retrieval": RETRIEVAL,
        "legacy_arm": "R5 RRF-free min-max 50/50 BM25 plus MiniLM hybrid, five units",
        "answer_judge_protocol": {
            "source": "R5 Stages 2 and 3", "temperature": 0.0, "enable_thinking": False,
            "workers": args.workers, "answer_request_count": len(benchmark.prompts),
            "shared_endpoint_for_automated_paraphrases_answering_judging": True,
        },
        "stress": {
            LME: {"n_paraphrases_per_item": 6, "n_injected_total": 6 * len(ku_items),
                  "source": "oldest answer-annotated session",
                  "timestamp": "strictly between the oldest and latest annotated sessions",
                  "paraphrase_input": "source transcript and marked user utterances only",
                  "n_unmarked_source_fallback": sum(
                      r["source_provenance"] != "annotated_user_span" for r in echo_by_id.values()
                  ),
                  "n_manually_paraphrased": sum(
                      r.get("generation_method") == "manual_source_only_paraphrase" for r in echo_by_id.values()
                  ),
                  "inherited_fixture_sha256": echo_meta.get("inherited_fixture_sha256"),
                  "echo_fixture": echo_path.name, "echo_fixture_sha256": sha256(echo_path)},
            LOCOMO: {"n_copies_per_source": 6, "n_sources_per_conversation": 15,
                     "n_injected_total": 6 * 15 * len(loc_convs),
                     "source": "MiniLM centroid proximity, no question or answer access, no header added"},
        },
        "csv_path": csv_path.name, "csv_sha256": sha256(csv_path),
        **results,
    }
    manifest_path = args.output / "r6_pm1b_manifest.json"
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(args.output / "report_receipt.json", {
        "schema_version": "r6_pm1b_report_receipt_v1",
        "artifact_hashes": {path.name: sha256(path) for path in (manifest_path, csv_path, echo_path)},
    })
    print(json.dumps(results["checks"], sort_keys=True), flush=True)
    print(f"R6 complete in {manifest['wall_seconds']}s, rows={len(rows)}, csv_sha256={manifest['csv_sha256']}", flush=True)
    if not results["checks"]["coverage_and_zero_api_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

