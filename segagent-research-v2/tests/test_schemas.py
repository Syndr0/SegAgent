from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.schemas import PlannerAction, PlannerDecision, SegmentRequest


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

if __name__ == "__main__":
    unittest.main()
