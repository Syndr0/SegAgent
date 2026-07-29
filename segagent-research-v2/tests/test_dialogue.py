import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dialogue import (
    apply_reply,
    guard_decision,
    has_spatial_detail,
    pre_plan_gate,
    rejection_notice,
    reply_sentiment,
)
from backend.schemas import PlannerAction, PlannerDecision

ORGANS = ("liver", "left kidney", "spinal cord")


def gtv_state(**kw) -> dict:
    state = {"intent": {"intent": "gtv", "confidence": 0.8}, "question": "contour the tumor"}
    state.update(kw)
    return state


def segment_decision() -> PlannerDecision:
    return PlannerDecision(
        action=PlannerAction.SEGMENT,
        rationale_summary="segment",
        confidence=0.9,
        structures=["parotid"],
    )


class SpatialDetailTests(unittest.TestCase):
    def test_bare_tumor_has_no_spatial_detail(self) -> None:
        self.assertFalse(has_spatial_detail("segment the tumor", ORGANS))

    def test_laterality_position_and_size_count(self) -> None:
        self.assertTrue(has_spatial_detail("the left upper lobe mass", ORGANS))
        self.assertTrue(has_spatial_detail("a 25 mm lesion", ORGANS))

    def test_naming_a_neighbouring_organ_counts(self) -> None:
        self.assertTrue(has_spatial_detail("the mass in the liver", ORGANS))

    def test_empty_text(self) -> None:
        self.assertFalse(has_spatial_detail("", ORGANS))

    def test_terms_match_on_word_boundaries(self) -> None:
        # "based" must not match "base", nor "nodule" the organ list.
        self.assertFalse(has_spatial_detail("based on the report, find the tumor", ORGANS))
        self.assertFalse(has_spatial_detail("contour the tumour", ORGANS))


class ReplySentimentTests(unittest.TestCase):
    def test_basic_moods(self) -> None:
        self.assertEqual(reply_sentiment("yes, that's right"), "yes")
        self.assertEqual(reply_sentiment("no, wrong place"), "no")
        self.assertEqual(reply_sentiment("I'm not sure"), "unknown")
        self.assertEqual(reply_sentiment("the left lobe"), "other")

    def test_negation_wins_over_contained_affirmative(self) -> None:
        # "not correct" contains "correct"; it must not read as agreement.
        self.assertEqual(reply_sentiment("that is not correct"), "no")


class GtvGateTests(unittest.TestCase):
    def test_asks_when_tumor_request_has_no_location(self) -> None:
        gate = pre_plan_gate(gtv_state(), ORGANS)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.action, PlannerAction.ASK_USER)
        self.assertEqual(gate.gate, "gtv_spatial")

    def test_does_not_ask_when_location_is_given(self) -> None:
        state = gtv_state(question="contour the mass in the left upper lobe")
        self.assertIsNone(pre_plan_gate(state, ORGANS))

    def test_does_not_ask_twice(self) -> None:
        state = gtv_state(gates_fired=["gtv_spatial"])
        self.assertIsNone(pre_plan_gate(state, ORGANS))

    def test_description_reply_enriches_the_prompt(self) -> None:
        state = gtv_state()
        update = apply_reply(state, "gtv_spatial", "it is in the left kidney", ORGANS)
        self.assertIn("left kidney", update["intent"]["source_description"])
        self.assertNotIn("gtv_blind_attempt", update["answers"])

    def test_unknown_reply_falls_back_to_a_blind_attempt(self) -> None:
        state = gtv_state()
        update = apply_reply(state, "gtv_spatial", "I'm not sure", ORGANS)
        self.assertTrue(update["answers"]["gtv_blind_attempt"])

    def test_confirm_gate_after_a_blind_attempt(self) -> None:
        state = gtv_state(
            gates_fired=["gtv_spatial"],
            answers={"gtv_blind_attempt": True, "gtv_first_mask": True},
        )
        gate = pre_plan_gate(state, ORGANS)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.gate, "gtv_confirm")

    def test_no_confirm_gate_before_a_contour_exists(self) -> None:
        # The blind attempt was requested but produced no mask yet.
        state = gtv_state(gates_fired=["gtv_spatial"], answers={"gtv_blind_attempt": True})
        self.assertIsNone(pre_plan_gate(state, ORGANS))

    def test_no_confirm_gate_once_the_location_is_settled(self) -> None:
        for settled in ("gtv_location_confirmed", "gtv_rejected"):
            state = gtv_state(
                gates_fired=["gtv_spatial"],
                answers={"gtv_blind_attempt": True, "gtv_first_mask": True, settled: True},
            )
            self.assertIsNone(pre_plan_gate(state, ORGANS), settled)

    def test_rejection_stops_the_run_instead_of_repeating(self) -> None:
        state = gtv_state()
        update = apply_reply(state, "gtv_confirm", "no, completely wrong", ORGANS)
        self.assertTrue(update["answers"]["gtv_rejected"])
        self.assertIsNotNone(rejection_notice({"answers": update["answers"]}))

    def test_confirmation_records_the_location(self) -> None:
        update = apply_reply(gtv_state(), "gtv_confirm", "yes that's right", ORGANS)
        self.assertTrue(update["answers"]["gtv_location_confirmed"])
        self.assertIsNone(rejection_notice({"answers": update["answers"]}))


class OarGuardTests(unittest.TestCase):
    def test_blocks_segmenting_without_a_confirmed_site(self) -> None:
        state = {"intent": {"intent": "oar", "confidence": 0.8}}
        guarded = guard_decision(state, segment_decision())
        self.assertEqual(guarded.action, PlannerAction.ASK_USER)
        self.assertEqual(guarded.gate, "oar_site")

    def test_allows_segmenting_once_the_site_is_known(self) -> None:
        state = {"intent": {"intent": "oar", "site": "thorax"}}
        self.assertEqual(guard_decision(state, segment_decision()).action, PlannerAction.SEGMENT)

    def test_allows_segmenting_after_the_user_confirms(self) -> None:
        state = {"intent": {"intent": "oar"}, "answers": {"site": "pelvis"}}
        self.assertEqual(guard_decision(state, segment_decision()).action, PlannerAction.SEGMENT)

    def test_reply_confirms_the_site(self) -> None:
        state = {"intent": {"intent": "oar"}}
        update = apply_reply(state, "oar_site", "thorax", ORGANS)
        self.assertEqual(update["intent"]["site"], "thorax")
        self.assertEqual(update["answers"]["site"], "thorax")

    def test_plain_organ_request_is_not_guarded(self) -> None:
        state = {"intent": {"intent": "organ", "targets": ["liver"]}}
        self.assertEqual(guard_decision(state, segment_decision()).action, PlannerAction.SEGMENT)


if __name__ == "__main__":
    unittest.main()
