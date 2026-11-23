"""Regression coverage for dated retrieval, conservative gates, and R5 parity."""

import unittest
from unittest.mock import patch

import numpy as np

from chelated_conversational_v3 import (
    BGE_DIMENSIONS,
    EARLIER,
    LATEST,
    ChelatedConversationalV3,
    ConversationTurn,
)
from scripts.run_r6_pm1b_conversational_sota_closure import (
    LME,
    LOCOMO,
    Question,
    _echo_source,
    lme_memory,
    make_lme_echoes,
    parse_echoes,
    r5_grade,
)


class FakeBGE:
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        vectors = np.zeros((len(texts), BGE_DIMENSIONS), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if "greenhouse" in text.lower() else 1] = 1.0
        return vectors


def turn(uid, sid, day, claim, *, speaker="User", position=0):
    return ConversationTurn(uid, sid, f"2024-03-{day:02d}", day * 86400 + position,
                            f"{speaker}: {claim}", claim, speaker)


class ConversationalRetrievalTests(unittest.TestCase):
    def test_clean_ranking_and_context_are_identical_and_expanded(self):
        encoder = FakeBGE()
        turns = [
            turn("neighbor", "first", 1, "The labels are in the bottom drawer."),
            turn("hit", "first", 1, "The greenhouse watering chart is laminated.", position=1),
            turn("other", "second", 2, "We buy apples on Saturday."),
        ]
        engine = ChelatedConversationalV3(turns, encoder)
        args = {"anchor_count": 1, "window_radius": 1}
        clean = engine.retrieve("Where is the greenhouse chart?", epistemic=False, **args)
        gated = engine.retrieve("Where is the greenhouse chart?", **args)
        self.assertEqual(clean, gated)
        self.assertEqual(gated.ranked_turn_ids, ("hit",))
        self.assertEqual([r.turn.turn_id for r in gated.turns], ["neighbor", "hit"])
        self.assertTrue(all(t.startswith("[Session Date: ") for t in encoder.texts[:3]))
        self.assertIn("[Session Date: 2024-03-01] User: The labels", gated.context)
        bounded = engine.retrieve("greenhouse chart", context_chars=len(gated.turns[-1].render()), **args)
        self.assertEqual([r.turn.turn_id for r in bounded.turns], ["hit"])

    def test_echo_representative_is_earliest_and_conflicting_value_remains(self):
        earlier = turn("old", "a", 1, "I set my daily greenhouse irrigation target to 20 liters.")
        duplicate = turn("echo", "b", 2, "My daily greenhouse irrigation target is set to 20 liters.")
        current = turn("new", "c", 3, "I changed my daily greenhouse irrigation target to 25 liters.")
        engine = ChelatedConversationalV3([duplicate, earlier, current], FakeBGE())
        result = engine.retrieve("What is the daily greenhouse irrigation target?")
        self.assertEqual(result.echo_clusters, (("old", "echo"),))
        self.assertEqual(result.supersessions, (("new", "old"),))
        self.assertEqual({t.turn.turn_id: t.status for t in result.turns}, {"old": EARLIER, "new": LATEST})

    def test_complete_linkage_cannot_collapse_transitive_chain(self):
        turns = [
            turn("a", "a", 1, "The greenhouse has violet plants along the west wall."),
            turn("b", "b", 2, "The greenhouse has flowers along the west wall."),
            turn("c", "c", 3, "The greenhouse has white plants along the west wall."),
        ]

        def classify(a, b):
            if "violet" in a + b and "white" in a + b:
                return "PRESERVE_CONFLICT", "CONFLICT_PROPER_ENTITY"
            return "ECHO_COLLAPSE", "COMPATIBLE_ECHO"

        engine = ChelatedConversationalV3(turns, FakeBGE())
        with patch("chelated_conversational_v3.classify_high_similarity_pair", side_effect=classify):
            result = engine.retrieve("greenhouse west wall", ranked=(0, 1, 2))
        self.assertEqual(result.echo_clusters, (("a", "b"),))
        self.assertEqual(set(result.ranked_turn_ids), {"a", "c"})

    def test_no_echo_or_update_edges_between_speakers_or_within_a_session(self):
        fact = "I changed my greenhouse irrigation timer to the evening."
        original = turn("a", "one", 1, fact)
        same = turn("b", "one", 1, fact, position=1)
        other = turn("c", "two", 2, fact, speaker="Neighbor")
        engine = ChelatedConversationalV3([original, same, other], FakeBGE())
        self.assertEqual(engine.retrieve("greenhouse"), engine.retrieve("greenhouse", epistemic=False))

    def test_empty_index_and_duplicate_ids(self):
        self.assertEqual(ChelatedConversationalV3([], FakeBGE()).retrieve("nothing").context, "")
        example = turn("a", "one", 1, "one record")
        with self.assertRaisesRegex(ValueError, "unique"):
            ChelatedConversationalV3([example, example], FakeBGE())


class ClosureRunnerTests(unittest.TestCase):
    def test_longmemeval_turn_pair_keeps_original_full_text_for_context(self):
        full_reply = "continued reply " * 100
        item = {
            "haystack_session_ids": ["a"], "haystack_dates": ["2023/06/25 (Sun) 13:22"],
            "haystack_sessions": [[
                {"role": "user", "content": "Where is the calendar?"},
                {"role": "assistant", "content": full_reply},
            ]],
        }
        recalled, = lme_memory(item)
        self.assertIn(full_reply.strip(), recalled.text)
        self.assertLess(len(recalled.indexed_text), len(recalled.text))
        self.assertTrue(recalled.indexed_text.startswith("[Session Date: 2023/06/25 (Sun) 13:22]"))

    def test_unmarked_stress_source_uses_only_older_user_text(self):
        earlier = "I drink two cups of tea at breakfast."
        item = {
            "question_id": "example_abs", "answer_session_ids": ["old", "new"],
            "haystack_session_ids": ["old", "new"],
            "haystack_dates": ["2023/06/25 (Sun) 13:22", "2023/06/26 (Mon) 13:22"],
            "haystack_sessions": [
                [{"role": "user", "content": earlier}, {"role": "assistant", "content": "Enjoy."}],
                [{"role": "user", "content": "I drink coffee now.", "has_answer": True}],
            ],
        }
        before, after, sid, dialogue, marked, source = _echo_source(item)
        self.assertLess(before, after)
        self.assertEqual((sid, marked, source), ("old", [earlier], "unmarked_earlier_session_user_turn"))
        self.assertNotIn("coffee", dialogue)

    def test_paraphrase_fixture_rejects_duplicates_and_bare_copies(self):
        with self.assertRaises(ValueError):
            parse_echoes('["I drink tea at breakfast"]', [])
        with self.assertRaises(ValueError):
            parse_echoes('["I drink tea at breakfast", "I enjoy morning tea", "Tea is my morning drink", '
                         '"I brew tea each morning", "I take tea with breakfast", "I have tea after waking"]',
                         ["I drink tea at breakfast"])

    def test_paraphrase_array_failure_falls_back_to_distinct_utterances(self):
        item = {
            "question_id": "ku", "answer_session_ids": ["a", "b"], "haystack_session_ids": ["a", "b"],
            "haystack_dates": ["2023/06/25 (Sun) 13:22", "2023/06/26 (Mon) 13:22"],
            "haystack_sessions": [
                [{"role": "user", "content": "I take the bus to work.", "has_answer": True}],
                [{"role": "user", "content": "I walk to work now.", "has_answer": True}],
            ],
        }
        answers = iter([
            '[]', '[]', 'I commute by bus every day.', 'The bus is how I get to work.',
            'I catch the bus for work.', 'My work commute is by bus.',
            'I ride the bus to my job.', 'I get to my workplace by bus.',
        ])
        fixture = make_lme_echoes(item, lambda *args: (next(answers), 0, 0.0, False))
        self.assertEqual(len(fixture["turns"]), 6)
        self.assertTrue(all("bus" in entry["claim_text"] for entry in fixture["turns"]))

    def test_valid_array_followed_by_model_commentary(self):
        echoes = parse_echoes(
            '["I enjoy morning tea", "Tea is my morning drink", "I brew tea each morning", '
            '"I take tea with breakfast", "I have tea after waking", "My mornings start with tea"]'
            "\nThose are six options.",
            [],
        )
        self.assertEqual(len(echoes), 6)
        self.assertEqual(echoes[-1], "My mornings start with tea")

    def test_r5_lme_shortcut_and_locomo_refusal(self):
        lme = Question(LME, "a", "knowledge-update", "Which city?", "Berlin")
        self.assertIs(r5_grade(lme, "It's Berlin."), True)
        self.assertIn("Reference Answer: Berlin", r5_grade(lme, "Munich")[1])
        loc = Question(LOCOMO, "b", "5", "What color?", "UNANSWERABLE", abstention=True)
        self.assertIs(r5_grade(loc, "Not mentioned in the dialogue."), True)
        self.assertIs(r5_grade(loc, "Green"), False)

    def test_j4_speaker_header_and_holdout_split_are_deterministic_and_non_forcing(self):
        from scripts.run_r6_pm1b_conversational_sota_closure import build_lme_holdout_split_map, question_split_tag

        turns = [
            ConversationTurn("t1", "s1", "2024-03-01", 100.0, "Alice: I bought a red kayak.", "I bought a red kayak.", "Alice"),
            ConversationTurn("t2", "s2", "2024-03-02", 200.0, "Bob: I love hiking in the Alps.", "I love hiking in the Alps.", "Bob"),
        ]
        engine = ChelatedConversationalV3(turns, FakeBGE())
        rec = engine.retrieve(
            "What kayak did Bob buy?",
            speaker_attribution_header=True,
            speaker_anchor_reserve=1,
            adaptive_rrf_gap=0.14,
            min_sessions=1,
            max_sessions=2,
            session_order="relevance",
        )
        self.assertIn("Question Subject: Bob", rec.header_line)
        self.assertNotIn("UNANSWERABLE", rec.header_line)
        self.assertTrue(rec.context.startswith("[Dialogue Cast: Alice, Bob"))

        items = [{"question_id": f"q_{i}", "question_type": "knowledge-update"} for i in range(10)]
        smap1 = build_lme_holdout_split_map(items)
        smap2 = build_lme_holdout_split_map(items)
        self.assertEqual(smap1, smap2)
        self.assertEqual(sum(v == "DEV" for v in smap1.values()), 5)
        self.assertEqual(sum(v == "HELD_OUT_TEST" for v in smap1.values()), 5)
        self.assertEqual(question_split_tag(Question(LOCOMO, "loc_4_0", "1", "Q?", "A", group=4), smap1), "DEV")
        self.assertEqual(question_split_tag(Question(LOCOMO, "loc_5_0", "1", "Q?", "A", group=5), smap1), "HELD_OUT_TEST")



if __name__ == "__main__":
    unittest.main()
