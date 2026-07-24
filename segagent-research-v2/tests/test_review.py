from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.schemas import ApprovalDecision, ApprovalKind


class ApprovalDecisionTests(unittest.TestCase):
    def test_modify_requires_edited_mask_id(self) -> None:
        with self.assertRaises(ValidationError):
            ApprovalDecision(decision="modify")

    def test_modify_with_mask_id_is_valid(self) -> None:
        decision = ApprovalDecision(decision="modify", edited_mask_id="artifact_abc123")
        self.assertEqual(decision.decision, ApprovalKind.MODIFY)
        self.assertEqual(decision.edited_mask_id, "artifact_abc123")

    def test_feedback_still_requires_text(self) -> None:
        with self.assertRaises(ValidationError):
            ApprovalDecision(decision="feedback", feedback="   ")

    def test_approve_needs_nothing_extra(self) -> None:
        self.assertEqual(ApprovalDecision(decision="approve").decision, ApprovalKind.APPROVE)


if __name__ == "__main__":
    unittest.main()
