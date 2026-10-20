from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from backend.dialogue import gtv_spatial_gate, required_workflow_decision
from backend.planner import QwenStructuredPlanner
from backend.prompts import (
    clean_prompt,
    fallback_prompt_normalizations,
    split_prompt_sources,
    trusted_prompt_normalizations,
    validate_prompt_normalizations,
)
from backend.schemas import (
    PlannerAction,
    PromptCategory,
    PromptNormalization,
    PromptNormalizationBatch,
    SegmentRequest,
)

WORKFLOW_IMPORT_ERROR = ""
try:
    from backend.workflow import SegAgentGraph
except ModuleNotFoundError as exc:  # optional workflow deps may be absent
    if exc.name not in {
        "langgraph",
        "langgraph.checkpoint",
        "langgraph.checkpoint.sqlite",
        "pydantic_settings",
    }:
        raise
    SegAgentGraph = None
    WORKFLOW_IMPORT_ERROR = exc.name or str(exc)


def item(
    index: int,
    source: str,
    category: PromptCategory,
    prompt: str | None = None,
) -> PromptNormalization:
    return PromptNormalization(
        source_index=index,
        source_text=source,
        category=category,
        voxtell_prompt=prompt or source,
        confidence=0.9,
    )


class PromptNormalizationPolicyTests(unittest.TestCase):
    def test_blind_carcinoma_prompt_is_explicitly_exploratory(self) -> None:
        batch = trusted_prompt_normalizations(
            ["carcinoma"],
            PromptCategory.PATHOLOGICAL_STRUCTURE,
            source_texts=["tumor"],
            exploratory=True,
            reason="Exploratory blind report-language prompt; not a diagnosis",
        )

        normalization = batch.items[0]
        self.assertEqual(normalization.source_text, "tumor")
        self.assertEqual(normalization.voxtell_prompt, "carcinoma")
        self.assertEqual(normalization.normalized_by, "trusted_workflow")
        self.assertTrue(normalization.exploratory)
        self.assertEqual(
            normalization.fallback_reason,
            "Exploratory blind report-language prompt; not a diagnosis",
        )

    def test_regular_normalization_never_globally_rewrites_tumor(self) -> None:
        normalization = fallback_prompt_normalizations(["tumor"]).items[0]

        self.assertEqual(normalization.source_text, "tumor")
        self.assertEqual(normalization.voxtell_prompt, "tumor")
        self.assertFalse(normalization.exploratory)

    def test_user_validated_prompts_keep_their_voxtell_wording(self) -> None:
        sources = [
            "liver",
            "stomach",
            "left kidney",
            "pancreas",
            "skin",
            "small intestine",
            "gallbladder",
        ]
        proposed = PromptNormalizationBatch(
            items=[
                item(
                    index,
                    source,
                    PromptCategory.ANATOMICAL_CONCEPT,
                    "small bowel" if source == "small intestine" else source,
                )
                for index, source in enumerate(sources)
            ]
        )

        normalized = validate_prompt_normalizations(sources, proposed)

        self.assertEqual(
            [entry.voxtell_prompt for entry in normalized.items],
            sources,
        )

    def test_dicom_eye_aliases_become_natural_language(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(0, "eye_L", PromptCategory.ANATOMICAL_CONCEPT, "left eye"),
                item(1, "eye_R", PromptCategory.ANATOMICAL_CONCEPT, "right eye"),
            ]
        )

        normalized = validate_prompt_normalizations(["eye_L", "eye_R"], batch)

        self.assertEqual(
            [entry.voxtell_prompt for entry in normalized.items],
            ["left eye", "right eye"],
        )

    def test_pathology_and_report_language_cannot_be_paraphrased(self) -> None:
        report = "Bronchial carcinoma in the right upper lobe with pleural contact"
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "bladder carcinoma",
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                    "bladder tumor",
                ),
                item(
                    1,
                    report,
                    PromptCategory.CLINICAL_DESCRIPTION,
                    "right lung cancer touching pleura",
                ),
            ]
        )

        normalized = validate_prompt_normalizations(
            ["bladder carcinoma", report], batch
        )

        self.assertEqual(normalized.items[0].voxtell_prompt, "bladder carcinoma")
        self.assertEqual(normalized.items[1].voxtell_prompt, report)

    def test_anatomy_normalization_cannot_flip_laterality(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "left optic nerve",
                    PromptCategory.ANATOMICAL_CONCEPT,
                    "right optic nerve",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "laterality"):
            validate_prompt_normalizations(["left optic nerve"], batch)

    def test_llm_cannot_substitute_one_anatomical_concept_for_another(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "spleen",
                    PromptCategory.ANATOMICAL_CONCEPT,
                    "liver",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "different anatomical concept"):
            validate_prompt_normalizations(["spleen"], batch)

    def test_pathology_cannot_be_downgraded_to_anatomy(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "renal cell carcinoma",
                    PromptCategory.ANATOMICAL_CONCEPT,
                    "kidney",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "wrong prompt category"):
            validate_prompt_normalizations(["renal cell carcinoma"], batch)

    def test_known_protocol_anatomy_cannot_be_misclassified(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "left optic nerve",
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "known anatomical"):
            validate_prompt_normalizations(
                ["left optic nerve"],
                batch,
                known_anatomy=["left optic nerve", "heart"],
            )

    def test_anatomical_alias_cannot_be_misclassified(self) -> None:
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "eye_L",
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "known anatomical"):
            validate_prompt_normalizations(["eye_L"], batch)

    def test_fallback_covers_all_three_categories_and_typo_alias(self) -> None:
        report = "Bronchial carcinoma in the right upper lobe with pleural contact"

        batch = fallback_prompt_normalizations(
            ["iver", "bladder carcinoma", report],
            reason="synthetic failure",
        )

        self.assertEqual(
            [entry.category for entry in batch.items],
            [
                PromptCategory.ANATOMICAL_CONCEPT,
                PromptCategory.PATHOLOGICAL_STRUCTURE,
                PromptCategory.CLINICAL_DESCRIPTION,
            ],
        )
        self.assertEqual(
            [entry.voxtell_prompt for entry in batch.items],
            ["liver", "bladder carcinoma", report],
        )
        self.assertTrue(
            all(entry.normalized_by == "deterministic_fallback" for entry in batch.items)
        )


class PromptSourceExtractionTests(unittest.TestCase):
    def test_request_prefix_never_consumes_anatomy_name_prefixes(self) -> None:
        for source in (
            "appendix",
            "aorta",
            "adrenal gland",
            "adenoma",
            "anal canal",
            "thecal sac",
        ):
            with self.subTest(source=source):
                self.assertEqual(clean_prompt(f"segment {source}"), source)

    def test_user_validated_mixed_example_has_three_typed_regimes(self) -> None:
        report = "Bronchial carcinoma in the right upper lobe with pleural contact"
        sources = split_prompt_sources(
            "iver, stomach, left kidney, pancreas, skin, small intestine, "
            f"gallbladder, bladder carcinoma, {report}"
        )

        batch = fallback_prompt_normalizations(sources)

        self.assertEqual(
            [entry.voxtell_prompt for entry in batch.items],
            [
                "liver",
                "stomach",
                "left kidney",
                "pancreas",
                "skin",
                "small intestine",
                "gallbladder",
                "bladder carcinoma",
                report,
            ],
        )
        self.assertEqual(
            [entry.category for entry in batch.items],
            [
                *([PromptCategory.ANATOMICAL_CONCEPT] * 7),
                PromptCategory.PATHOLOGICAL_STRUCTURE,
                PromptCategory.CLINICAL_DESCRIPTION,
            ],
        )

    def test_comma_separated_known_and_open_anatomy_are_independent(self) -> None:
        self.assertEqual(
            split_prompt_sources(
                "segment liver, skin, small intestine",
                known_anatomy=["liver"],
            ),
            ["liver", "skin", "small intestine"],
        )

    def test_and_separated_eye_aliases_and_anatomy_are_independent(self) -> None:
        self.assertEqual(
            split_prompt_sources(
                "segment eye_L and liver",
                known_anatomy=["liver"],
            ),
            ["eye_L", "liver"],
        )
        self.assertEqual(
            split_prompt_sources("segment eye_L and eye_R"),
            ["eye_L", "eye_R"],
        )

    def test_report_commas_are_not_split_from_continuations(self) -> None:
        report = (
            "Bronchial carcinoma in the right upper lobe, "
            "with pleural contact"
        )
        self.assertEqual(split_prompt_sources(f"segment {report}"), [report])

    def test_report_modifiers_and_parentheses_are_not_fragmented(self) -> None:
        reports = [
            "irregular, spiculated mass in the right upper lobe",
            "Bronchial carcinoma in the right upper lobe, pleural contact",
            "Bronchial carcinoma, right upper lobe, pleural contact",
            "mass (right upper lobe, pleural contact)",
            "mass involving liver, stomach and pancreas",
            "Bronchial carcinoma in the right upper lobe; with pleural contact",
        ]
        for report in reports:
            with self.subTest(report=report):
                self.assertEqual(split_prompt_sources(report), [report])

    def test_oxford_comma_anatomy_list_is_split_cleanly(self) -> None:
        self.assertEqual(
            split_prompt_sources(
                "segment liver, stomach, and pancreas",
                known_anatomy=["liver", "stomach", "pancreas"],
            ),
            ["liver", "stomach", "pancreas"],
        )

    def test_pathology_and_report_items_in_a_list_are_split(self) -> None:
        report = "Bronchial carcinoma in the right upper lobe with pleural contact"
        self.assertEqual(
            split_prompt_sources(f"bladder carcinoma, {report}"),
            ["bladder carcinoma", report],
        )


class QwenPromptNormalizerTests(unittest.TestCase):
    def test_text_only_structured_result_is_validated(self) -> None:
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=SimpleNamespace(),
            enable_thinking=True,
        )
        payload = {
            "items": [
                {
                    "source_index": 0,
                    "source_text": "eye_L",
                    "category": "anatomical_concept",
                    "voxtell_prompt": "left eye",
                    "confidence": 0.96,
                    "exploratory": True,
                }
            ]
        }
        raw = f"<think>Resolve the side suffix.</think>{json.dumps(payload)}"

        with (
            patch.object(planner, "_ensure"),
            patch.object(planner, "_generate", return_value=raw) as generate,
        ):
            result = planner.normalize_prompts(["eye_L"])

        self.assertEqual(result.items[0].voxtell_prompt, "left eye")
        self.assertEqual(result.thinking, "Resolve the side suffix.")
        self.assertFalse(result.items[0].exploratory)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 768)

    def test_invalid_batch_cardinality_is_rejected_after_repair_attempt(self) -> None:
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=SimpleNamespace(),
        )
        malformed = json.dumps({"items": []})

        with (
            patch.object(planner, "_ensure"),
            patch.object(planner, "_generate", return_value=malformed) as generate,
        ):
            with self.assertRaisesRegex(RuntimeError, "prompt normalizer failed"):
                planner.normalize_prompts(["left kidney"])

        self.assertEqual(generate.call_count, 2)


@unittest.skipIf(
    SegAgentGraph is None,
    f"workflow dependency is not installed: {WORKFLOW_IMPORT_ERROR}",
)
class WorkflowPromptNormalizationTests(unittest.TestCase):
    def test_normalizer_failure_falls_back_without_blocking_segmentation(self) -> None:
        class BrokenNormalizer:
            def normalize_prompts(self, prompts, on_thinking_delta=None):
                del prompts, on_thinking_delta
                raise RuntimeError("bad structured output")

        graph = object.__new__(SegAgentGraph)
        graph.planner = BrokenNormalizer()

        batch, _, events = graph._normalize_prompt_batch(
            ["eye_L"], cache={}, step=1
        )

        self.assertEqual(batch.items[0].voxtell_prompt, "left eye")
        self.assertEqual(batch.items[0].normalized_by, "deterministic_fallback")
        self.assertTrue(events[-1]["payload"]["fallback"])

    def test_pathology_category_refines_open_vocabulary_intent(self) -> None:
        state = {
            "intent": {
                "intent": "organ",
                "targets": ["schwannoma"],
                "rationale": "open target",
            }
        }
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "schwannoma",
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                )
            ]
        )

        SegAgentGraph._apply_prompt_categories(state, batch)

        self.assertEqual(state["intent"]["intent"], "gtv")
        self.assertEqual(state["intent"]["source_description"], "schwannoma")

    def test_clinical_category_refines_open_vocabulary_intent(self) -> None:
        state = {
            "intent": {
                "intent": "organ",
                "targets": ["pelvic wall"],
                "rationale": "open target",
            }
        }
        batch = PromptNormalizationBatch(
            items=[
                item(
                    0,
                    "soft tissue thickening along the pelvic wall",
                    PromptCategory.CLINICAL_DESCRIPTION,
                )
            ]
        )

        SegAgentGraph._apply_prompt_categories(state, batch)

        self.assertEqual(state["intent"]["intent"], "gtv")
        self.assertEqual(
            state["intent"]["source_description"],
            "soft tissue thickening along the pelvic wall",
        )

    def test_raw_open_target_survives_known_anatomical_anchor(self) -> None:
        graph = object.__new__(SegAgentGraph)
        graph._organ_names = ("aorta",)
        state = {
            "question": "segment schwannoma adjacent to the aorta",
            "intent": {
                "intent": "organ",
                "targets": ["aorta"],
            },
        }

        self.assertEqual(
            graph._normalization_sources(state),
            ["schwannoma adjacent to the aorta"],
        )

    def test_raw_list_keeps_open_anatomy_items(self) -> None:
        graph = object.__new__(SegAgentGraph)
        graph._organ_names = ("liver",)
        state = {
            "question": "segment liver, skin, small intestine",
            "intent": {
                "intent": "organ",
                "targets": ["liver"],
            },
        }

        self.assertEqual(
            graph._normalization_sources(state),
            ["liver", "skin", "small intestine"],
        )


class TumorCategoryRoutingTests(unittest.TestCase):
    def test_named_pathology_bypasses_generic_spatial_question(self) -> None:
        state = {
            "question": "segment bladder carcinoma",
            "intent": {
                "intent": "gtv",
                "source_description": "bladder carcinoma",
            },
            "answers": {},
            "prompt_normalizations": [
                item(
                    0,
                    "bladder carcinoma",
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                ).model_dump(mode="json")
            ],
        }

        self.assertIsNone(gtv_spatial_gate(state))
        decision = required_workflow_decision(state)
        self.assertEqual(decision.action, PlannerAction.SEGMENT)
        self.assertEqual(decision.structures, ["bladder carcinoma"])

    def test_generic_bare_tumor_still_requests_spatial_context(self) -> None:
        for source in ("tumor", "carcinoma"):
            with self.subTest(source=source):
                state = {
                    "question": f"segment {source}",
                    "intent": {
                        "intent": "gtv",
                        "source_description": source,
                    },
                    "answers": {},
                    "prompt_normalizations": [
                        item(
                            0,
                            source,
                            PromptCategory.PATHOLOGICAL_STRUCTURE,
                        ).model_dump(mode="json")
                    ],
                }

                decision = gtv_spatial_gate(state)

                self.assertIsNotNone(decision)
                self.assertEqual(decision.action, PlannerAction.ASK_USER)

    def test_generic_tumor_cannot_bypass_gate_via_clinical_category(self) -> None:
        state = {
            "question": "segment tumor",
            "intent": {"intent": "gtv", "source_description": "tumor"},
            "answers": {},
            "prompt_normalizations": [
                item(
                    0,
                    "tumor",
                    PromptCategory.CLINICAL_DESCRIPTION,
                ).model_dump(mode="json")
            ],
        }

        decision = gtv_spatial_gate(state)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, PlannerAction.ASK_USER)

    def test_numbered_generic_tumor_still_requests_spatial_context(self) -> None:
        for source in (
            "tumor 2",
            "tumor-2",
            "tumor II",
            "tumor A",
            "mass 3",
            "lesion A",
            "carcinoma no. 2",
            "multiple lesions",
            "possible mass",
            "suspected carcinoma",
            "recurrent tumor",
            "GTV",
            "GTV2",
            "CTV 1",
            "PTV",
        ):
            with self.subTest(source=source):
                state = {
                    "question": f"segment {source}",
                    "intent": {
                        "intent": "gtv",
                        "source_description": source,
                    },
                    "answers": {},
                    "prompt_normalizations": [
                        item(
                            0,
                            source,
                            PromptCategory.PATHOLOGICAL_STRUCTURE,
                        ).model_dump(mode="json")
                    ],
                }

                decision = gtv_spatial_gate(state)

                self.assertIsNotNone(decision)
                self.assertEqual(decision.action, PlannerAction.ASK_USER)


class PromptNormalizationSchemaTests(unittest.TestCase):
    def test_batch_requires_each_source_index_once_in_order(self) -> None:
        for indexes in ([9], [0, 0], [1, 0]):
            with self.subTest(indexes=indexes):
                with self.assertRaises(ValidationError):
                    PromptNormalizationBatch(
                        items=[
                            item(
                                index,
                                f"source {position}",
                                PromptCategory.ANATOMICAL_CONCEPT,
                            )
                            for position, index in enumerate(indexes)
                        ]
                    )

    def test_segment_request_requires_one_to_one_prompt_provenance(self) -> None:
        normalization = item(
            0,
            "eye_L",
            PromptCategory.ANATOMICAL_CONCEPT,
            "left eye",
        )
        request = SegmentRequest(
            case_id="case_x",
            structures=["left eye"],
            prompt_normalizations=[normalization],
        )
        self.assertEqual(request.prompt_normalizations[0].source_text, "eye_L")

        with self.assertRaises(ValidationError):
            SegmentRequest(
                case_id="case_x",
                structures=["right eye"],
                prompt_normalizations=[normalization],
            )

    def test_segment_request_requires_sequential_source_indexes(self) -> None:
        normalization = item(
            9,
            "left eye",
            PromptCategory.ANATOMICAL_CONCEPT,
        )

        with self.assertRaises(ValidationError):
            SegmentRequest(
                case_id="case_x",
                structures=["left eye"],
                target_type="organ",
                prompt_normalizations=[normalization],
            )

    def test_organ_request_rejects_pathological_prompt_category(self) -> None:
        normalization = item(
            0,
            "bladder carcinoma",
            PromptCategory.PATHOLOGICAL_STRUCTURE,
        )

        with self.assertRaises(ValidationError):
            SegmentRequest(
                case_id="case_x",
                structures=["bladder carcinoma"],
                target_type="organ",
                prompt_normalizations=[normalization],
            )

    def test_gtv_request_rejects_anatomical_prompt_category(self) -> None:
        normalization = item(
            0,
            "left kidney",
            PromptCategory.ANATOMICAL_CONCEPT,
        )

        with self.assertRaises(ValidationError):
            SegmentRequest(
                case_id="case_x",
                structures=["left kidney"],
                target_type="gtv",
                prompt_normalizations=[normalization],
            )


if __name__ == "__main__":
    unittest.main()
