"""Wiring tests: real classifier output must actually drive the gates.

The unit tests for dialogue.py feed hand-built intent dicts, so they kept
passing while classify() and its caller disagreed about an argument and every
gate silently received an empty intent. These tests go end to end from the
question text to the gate decision.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dialogue import guard_decision, pre_plan_gate
from backend.intent import IntentClassifier
from backend.knowledge import HybridKnowledgeBase
from backend.schemas import Intent, PlannerAction, PlannerDecision

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


def segment_decision(*structures: str) -> PlannerDecision:
    return PlannerDecision(
        action=PlannerAction.SEGMENT,
        rationale_summary="segment",
        confidence=0.9,
        structures=list(structures) or ["tumor"],
    )


class IntentDrivesGatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        knowledge = HybridKnowledgeBase(KNOWLEDGE)
        cls.classifier = IntentClassifier.from_knowledge(knowledge)
        cls.organs = tuple(
            organ for p in knowledge.protocols for organ in p.get("oars", [])
        )

    def state_for(self, question: str, has_contours: bool = False) -> dict:
        intent = self.classifier.classify(question, has_contours=has_contours)
        return {"intent": intent.model_dump(mode="json"), "question": question}

    def test_classify_accepts_the_arguments_its_caller_passes(self) -> None:
        # The workflow calls classify(text, has_contours=...); a signature drift
        # here disables every gate downstream.
        intent = self.classifier.classify("check the contours", has_contours=True)
        self.assertEqual(intent.intent, Intent.QC)

    def test_bare_tumor_question_asks_for_a_location(self) -> None:
        gate = pre_plan_gate(self.state_for("where is the tumor"), self.organs, False)
        self.assertIsNotNone(gate, "a tumor request with no location must ask first")
        self.assertEqual(gate.gate, "gtv_spatial")
        self.assertEqual(gate.action, PlannerAction.ASK_USER)

    def test_located_tumor_question_does_not_ask(self) -> None:
        state = self.state_for("contour the tumor in the left parotid gland")
        self.assertIsNone(pre_plan_gate(state, self.organs, False))

    def test_oar_request_cannot_segment_before_the_site_is_confirmed(self) -> None:
        guarded = guard_decision(self.state_for("find oar"), segment_decision())
        self.assertEqual(guarded.action, PlannerAction.ASK_USER)
        self.assertEqual(guarded.gate, "oar_site")

    def test_oar_request_with_a_site_may_segment(self) -> None:
        state = self.state_for("contour the head and neck OARs")
        self.assertEqual(
            guard_decision(state, segment_decision()).action, PlannerAction.SEGMENT
        )

    def test_plain_question_is_answered_not_segmented_for(self) -> None:
        # "What is in this image?" is visual question answering: it must route to
        # a description, never to a protocol lookup or a segmentation run.
        state = self.state_for("what's in the image")
        self.assertEqual(state["intent"]["intent"], Intent.QUESTION.value)
        gate = pre_plan_gate(state, self.organs, False)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.action, PlannerAction.ANSWER_QUESTION)

    def test_question_is_answered_only_once_per_turn(self) -> None:
        state = self.state_for("what's in the image")
        state["gates_fired"] = ["vqa"]
        self.assertIsNone(pre_plan_gate(state, self.organs, False))

    def test_report_request_without_contours_asks_first(self) -> None:
        gate = pre_plan_gate(self.state_for("give me a report"), self.organs, False)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.gate, "report_scope")


if __name__ == "__main__":
    unittest.main()
