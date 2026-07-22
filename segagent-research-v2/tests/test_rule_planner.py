from __future__ import annotations

import unittest
from pathlib import Path

from backend.knowledge import HybridKnowledgeBase
from backend.planner import RulePlanner
from backend.schemas import PlannerAction, ToolObservation


KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "knowledge"


class RulePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RulePlanner(HybridKnowledgeBase(KNOWLEDGE_ROOT))

    def test_explicit_structures_route_to_segmentation(self) -> None:
        decision = self.planner.decide(
            "case_000000000000", "Please segment the left kidney and right kidney", 1, []
        )

        self.assertEqual(decision.action, PlannerAction.SEGMENT)
        self.assertEqual(decision.structures, ["right kidney", "left kidney"])

    def test_protocol_question_routes_to_lookup(self) -> None:
        question = "Which OARs are in the prostate protocol?"
        decision = self.planner.decide("case_000000000000", question, 1, [])

        self.assertEqual(decision.action, PlannerAction.LOOKUP_PROTOCOL)
        self.assertEqual(decision.site_query, question)

    def test_protocol_observation_routes_to_all_oars(self) -> None:
        observation = ToolObservation(
            observation_id="observation_000000000000",
            tool="lookup_protocol",
            summary="Protocol found.",
            data={"site": "Pelvis", "oars": ["bladder", "rectum"]},
        )
        decision = self.planner.decide(
            "case_000000000000", "Segment the prostate OARs", 2, [observation]
        )

        self.assertEqual(decision.action, PlannerAction.SEGMENT)
        self.assertEqual(decision.structures, ["bladder", "rectum"])

    def test_unsupported_question_requests_clarification(self) -> None:
        decision = self.planner.decide(
            "case_000000000000", "Tell me something useful", 1, []
        )

        self.assertEqual(decision.action, PlannerAction.ASK_USER)


if __name__ == "__main__":
    unittest.main()
