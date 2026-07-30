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

    def test_oar_without_site_or_targets_abstains(self) -> None:
        # A bare OAR request names neither a site nor structures; the classifier
        # still returns OAR (the workflow's deterministic gate then asks the user
        # which site to contour rather than guessing an OAR list).
        result = self.clf.classify("find the OAR")
        self.assertEqual(result.intent, Intent.OAR)
        self.assertIsNone(result.site)
        self.assertEqual(result.targets, [])

    def test_ambiguous_target_and_oar_abstains(self) -> None:
        result = self.clf.classify("segment the tumor and spare the parotid gland")
        self.assertEqual(result.intent, Intent.AMBIGUOUS)

    def test_empty_request_abstains(self) -> None:
        self.assertEqual(self.clf.classify("do the thing").intent, Intent.AMBIGUOUS)

    def test_longest_non_overlapping_anatomy_match(self) -> None:
        clf = IntentClassifier(organ_names=["brain", "brainstem"])
        result = clf.classify("segment the brainstem")
        self.assertEqual(result.intent, Intent.ORGAN)
        self.assertEqual(result.targets, ["brainstem"])

    def test_site_alias_does_not_override_direct_organ(self) -> None:
        clf = IntentClassifier(
            organ_names=["left kidney", "left lung"],
            site_aliases=["kidney", "lung", "thorax"],
        )
        self.assertEqual(
            clf.classify("segment the left kidney").intent, Intent.ORGAN
        )
        self.assertEqual(clf.classify("segment the left lung").intent, Intent.ORGAN)

    def test_open_vocabulary_and_chinese_structure_requests(self) -> None:
        clf = IntentClassifier(organ_names=["left eye"])
        self.assertEqual(clf.classify("segment the appendix").targets, ["appendix"])
        chinese = clf.classify("分割左眼")
        self.assertEqual(chinese.intent, Intent.ORGAN)
        self.assertEqual(chinese.targets, ["left eye"])

    def test_visual_question_with_anatomy_is_not_segmentation(self) -> None:
        result = self.clf.classify("where is the left kidney?")
        self.assertEqual(result.intent, Intent.VQA)
        self.assertEqual(result.targets, ["left kidney"])

    def test_unclassified_question_falls_back_to_vqa(self) -> None:
        self.assertEqual(self.clf.classify("what is this?").intent, Intent.VQA)

    def test_mask_follow_up_has_an_explicit_intent(self) -> None:
        self.assertEqual(
            self.clf.classify("how big is the selected mask?").intent,
            Intent.MASK_QUESTION,
        )

    def test_report_scope_wins_over_target_terms(self) -> None:
        result = self.clf.classify("summarize the GTV and OAR contours in a report")
        self.assertEqual(result.intent, Intent.REPORT)

    def test_gtv_profile_demands_editing(self) -> None:
        profile = profile_for(Intent.GTV)
        self.assertEqual(profile.prompt_mode, "sentence")
        self.assertEqual(profile.trust, "mandatory_edit")
        self.assertEqual(profile_for(Intent.OAR).knowledge, "oar_protocol")


if __name__ == "__main__":
    unittest.main()
