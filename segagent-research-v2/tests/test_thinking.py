import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.planner import split_thinking
from backend.schemas import PlannerAction, PlannerDecision


class SplitThinkingTests(unittest.TestCase):
    def test_reasoning_is_separated_from_payload(self) -> None:
        reasoning, payload = split_thinking(
            '<think>The user wants the liver.</think>{"action": "segment"}'
        )
        self.assertEqual(reasoning, "The user wants the liver.")
        self.assertEqual(payload, '{"action": "segment"}')

    def test_response_without_thinking_is_all_payload(self) -> None:
        reasoning, payload = split_thinking('{"action": "final"}')
        self.assertEqual(reasoning, "")
        self.assertEqual(payload, '{"action": "final"}')

    def test_unclosed_thinking_yields_no_payload(self) -> None:
        # A truncated generation must not be parsed as a decision.
        reasoning, payload = split_thinking("<think>still reasoning")
        self.assertEqual(reasoning, "still reasoning")
        self.assertEqual(payload, "")

    def test_empty_response(self) -> None:
        self.assertEqual(split_thinking(""), ("", ""))


class InspectMaskDecisionTests(unittest.TestCase):
    def test_inspect_mask_requires_structures(self) -> None:
        with self.assertRaises(ValueError):
            PlannerDecision(
                action=PlannerAction.INSPECT_MASK, rationale_summary="r", confidence=0.5
            )

    def test_inspect_mask_accepts_structures(self) -> None:
        decision = PlannerDecision(
            action=PlannerAction.INSPECT_MASK,
            rationale_summary="measure the existing contour",
            confidence=0.8,
            structures=["left eye"],
        )
        self.assertEqual(decision.structures, ["left eye"])
        self.assertIsNone(decision.thinking)


if __name__ == "__main__":
    unittest.main()
