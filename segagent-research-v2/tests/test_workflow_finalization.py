from __future__ import annotations

import unittest

from backend.schemas import PlannerAction, PlannerDecision, ToolObservation
from backend.tools import EvidenceCritic
try:
    from backend.workflow import SegAgentGraph
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("langgraph"):
        SegAgentGraph = None
        WORKFLOW_IMPORT_ERROR = str(exc)
    else:
        raise
else:
    WORKFLOW_IMPORT_ERROR = ""


@unittest.skipIf(
    SegAgentGraph is None,
    f"workflow dependency unavailable: {WORKFLOW_IMPORT_ERROR}",
)
class WorkflowFinalizationTests(unittest.TestCase):
    @staticmethod
    def graph() -> SegAgentGraph:
        graph = object.__new__(SegAgentGraph)
        graph.critic = EvidenceCritic()
        return graph

    def test_blocked_model_answer_falls_back_to_verified_tool_summary(self) -> None:
        observation = ToolObservation(
            observation_id="observation_000000000020",
            tool="inspect_mask",
            summary=(
                "Existing contour measurements: eye_L has 3965 voxels, "
                "16.335 mL volume, and mean intensity 24.499 HU."
            ),
            data={
                "segment_result": {
                    "model_name": "inspect_mask",
                    "measurements": [
                        {
                            "structure": "eye_L",
                            "found": True,
                            "voxels": 3965,
                            "volume_ml": 16.335,
                            "mean_intensity": 24.499,
                        }
                    ],
                }
            },
        )
        decision = PlannerDecision(
            action=PlannerAction.FINAL,
            rationale_summary="Summarize the contour.",
            confidence=0.9,
            final_answer=(
                "eye_L is 16.335 mL and is visible on axial slices 95 through 97."
            ),
        )
        result = self.graph()._finish(
            {
                "current_decision": decision.model_dump(mode="json"),
                "observations": [observation.model_dump(mode="json")],
            }
        )

        self.assertIn("16.335", result["final_answer"])
        self.assertNotIn("95", result["final_answer"])
        self.assertNotIn("97", result["final_answer"])
        self.assertNotIn("I cannot return", result["final_answer"])
        self.assertTrue(result["critic_findings"])
        answer_event = next(
            item for item in result["events"] if item["type"] == "answer"
        )
        self.assertTrue(answer_event["payload"]["fallback_used"])
        self.assertFalse(
            any(
                item.severity == "error"
                for item in self.graph().critic.review(
                    result["final_answer"], [observation]
                )
            )
        )

    def test_no_evidence_still_returns_a_number_free_terminal_answer(self) -> None:
        decision = PlannerDecision(
            action=PlannerAction.FINAL,
            rationale_summary="Answer generated.",
            confidence=0.7,
            final_answer="The unmeasured volume is 999 mL.",
        )
        result = self.graph()._finish(
            {
                "current_decision": decision.model_dump(mode="json"),
                "observations": [],
            }
        )

        self.assertNotIn("999", result["final_answer"])
        self.assertNotIn("I cannot return", result["final_answer"])
        self.assertIn("Analysis completed", result["final_answer"])
        self.assertTrue(result["critic_findings"])
        self.assertEqual(result["events"][-1]["type"], "run_completed")


if __name__ == "__main__":
    unittest.main()
