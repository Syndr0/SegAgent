from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.prompts import build_prompts, target_type_for
from backend.schemas import Intent, SegmentRequest, TargetType, TaskIntent


class PromptBuilderTests(unittest.TestCase):
    def test_oar_intent_uses_labels_unchanged(self) -> None:
        intent = TaskIntent(intent=Intent.OAR, confidence=0.8, targets=["left kidney"])
        self.assertEqual(
            build_prompts(intent, ["left kidney", "liver"]), ["left kidney", "liver"]
        )

    def test_gtv_intent_uses_verbatim_sentence(self) -> None:
        sentence = "Bronchial carcinoma in the right upper lobe with pleural contact."
        intent = TaskIntent(intent=Intent.GTV, confidence=0.8, source_description=sentence)
        self.assertEqual(build_prompts(intent, []), [sentence])

    def test_gtv_without_description_falls_back_to_structures(self) -> None:
        intent = TaskIntent(intent=Intent.GTV, confidence=0.8)
        self.assertEqual(build_prompts(intent, ["mass"]), ["mass"])

    def test_no_intent_passes_structures_through(self) -> None:
        self.assertEqual(build_prompts(None, [" heart ", "heart"]), ["heart"])

    def test_target_type_mapping(self) -> None:
        self.assertEqual(
            target_type_for(TaskIntent(intent=Intent.GTV, confidence=1.0)), TargetType.GTV
        )
        self.assertEqual(
            target_type_for(TaskIntent(intent=Intent.OAR, confidence=1.0)), TargetType.OAR
        )
        self.assertEqual(target_type_for(None), TargetType.UNKNOWN)


class SegmentRequestTargetTests(unittest.TestCase):
    def test_target_type_defaults_to_unknown(self) -> None:
        request = SegmentRequest(case_id="case_x", structures=["liver"])
        self.assertEqual(request.target_type, TargetType.UNKNOWN)

    def test_long_clinical_sentence_prompt_is_accepted(self) -> None:
        sentence = (
            "Right retropharyngeal lymph node with suspected metastasis extending toward "
            "the carotid space, abutting the longus colli muscle on the involved side, with "
            "loss of the intervening fat plane and mild heterogeneous enhancement noted."
        )
        self.assertGreater(len(sentence), 160)
        request = SegmentRequest(
            case_id="case_x", structures=[sentence], target_type=TargetType.GTV
        )
        self.assertEqual(request.structures, [sentence])
        self.assertEqual(request.target_type, TargetType.GTV)

    def test_overlong_prompt_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SegmentRequest(case_id="case_x", structures=["a" * 401])


if __name__ == "__main__":
    unittest.main()
