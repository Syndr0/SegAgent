import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.planner import QwenStructuredPlanner, ThinkingDeltaParser, split_thinking
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

    def test_delta_parser_handles_split_tags_without_leaking_json(self) -> None:
        deltas = []
        parser = ThinkingDeltaParser(deltas.append)
        for chunk in ("<thi", "nk>first ", "thought</thi", "nk>{\"action\":\"final\"}"):
            parser.feed(chunk)
        self.assertEqual("".join(deltas), "first thought")
        self.assertTrue(parser.closed)

    def test_delta_parser_does_not_emit_non_thinking_prose(self) -> None:
        deltas = []
        parser = ThinkingDeltaParser(deltas.append)
        parser.feed("plain answer")
        self.assertEqual(deltas, [])


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


class ArtifactViewPriorityTests(unittest.TestCase):
    def test_named_mask_wins_over_many_ambient_visible_masks(self) -> None:
        artifacts = [
            SimpleNamespace(artifact_id=f"artifact_{index}", label=f"organ {index}", kind="mask")
            for index in range(10)
        ]
        artifacts.append(
            SimpleNamespace(artifact_id="artifact_eye", label="left eye", kind="mask")
        )
        case = SimpleNamespace(artifacts=artifacts)
        ids = QwenStructuredPlanner._artifact_view_ids(
            case,
            {
                "intent": {"targets": ["left eye"]},
                "message_context": {
                    "visible_mask_ids": [item.artifact_id for item in artifacts],
                },
            },
        )
        self.assertEqual(ids, ["artifact_eye"])


if __name__ == "__main__":
    unittest.main()
