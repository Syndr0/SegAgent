from __future__ import annotations

import unittest

from backend.intent import IntentClassifier, profile_for
from backend.schemas import Intent


class IntentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clf = IntentClassifier(
            organ_names=["liver", "left kidney", "right kidney", "parotid gland", "heart"],
            site_aliases=["thorax", "pelvis", "head and neck", "liver", "abdomen"],
        )

    def test_plain_organ_is_organ_not_oar(self) -> None:
        # "liver" is also a protocol alias, but a bare request stays ORGAN.
        result = self.clf.classify("segment the liver")
        self.assertEqual(result.intent, Intent.ORGAN)
        self.assertIn("liver", result.targets)

    def test_oar_from_region_and_keyword(self) -> None:
        self.assertEqual(self.clf.classify("contour the thorax OARs").intent, Intent.OAR)
        self.assertEqual(
            self.clf.classify("segment organs at risk for radiotherapy").intent, Intent.OAR
        )

    def test_gtv_from_tumor_language(self) -> None:
        result = self.clf.classify("segment the tumor in the liver")
        self.assertEqual(result.intent, Intent.GTV)
        self.assertIsNotNone(result.source_description)

    def test_gtv_from_report_sentence_is_passed_through(self) -> None:
        sentence = "Right retropharyngeal node, suspected metastasis."
        result = self.clf.classify("contour the described node", source_description=sentence)
        self.assertEqual(result.intent, Intent.GTV)
        self.assertEqual(result.source_description, sentence)

    def test_qc_requires_contours_else_abstains(self) -> None:
        self.assertEqual(
            self.clf.classify("check these contours", has_contours=True).intent, Intent.QC
        )
        self.assertEqual(
            self.clf.classify("check these contours", has_contours=False).intent, Intent.QUESTION
        )

    def test_ambiguous_target_and_oar_abstains(self) -> None:
        result = self.clf.classify("segment the tumor and spare the parotid gland")
        self.assertEqual(result.intent, Intent.QUESTION)

    def test_empty_request_abstains(self) -> None:
        self.assertEqual(self.clf.classify("do the thing").intent, Intent.QUESTION)

    def test_gtv_profile_demands_editing(self) -> None:
        profile = profile_for(Intent.GTV)
        self.assertEqual(profile.prompt_mode, "sentence")
        self.assertEqual(profile.trust, "mandatory_edit")
        self.assertEqual(profile_for(Intent.OAR).knowledge, "oar_protocol")


if __name__ == "__main__":
    unittest.main()
