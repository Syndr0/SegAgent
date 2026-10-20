from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.schemas import (
    MessageArtifactContext,
    PlannerAction,
    PlannerDecision,
    SegmentRequest,
)


class SchemaTests(unittest.TestCase):
    def test_segment_request_normalizes_and_deduplicates_names(self) -> None:
        request = SegmentRequest(
            case_id="case_000000000000",
            structures=[" left   kidney ", "Left Kidney", "right kidney"],
        )

        self.assertEqual(request.structures, ["left kidney", "right kidney"])

    def test_segment_decision_requires_structures(self) -> None:
        with self.assertRaises(ValidationError):
            PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="A target is required.",
                confidence=0.9,
            )

    def test_message_artifact_context_deduplicates_ids(self) -> None:
        context = MessageArtifactContext(
            referenced_artifact_ids=["artifact_a", "artifact_a"],
            visible_mask_ids=["artifact_b", "artifact_b"],
            selected_mask_id="artifact_b",
        )
        self.assertEqual(context.referenced_artifact_ids, ["artifact_a"])
        self.assertEqual(context.visible_mask_ids, ["artifact_b"])

if __name__ == "__main__":
    unittest.main()
