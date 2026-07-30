import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dialogue import pre_plan_gate
from backend.knowledge import HybridKnowledgeBase
from backend.qc import (
    ContourQCEngine,
    assess_report_readiness,
    canonical_roi,
    centroid_mm,
    dice,
    is_target,
    nearest_surface_distance_mm,
    target_class,
)
from backend.schemas import ArtifactRef, Intent, PlannerAction

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "knowledge"


class TargetClassificationTests(unittest.TestCase):
    def test_target_volumes_are_recognized(self) -> None:
        for label in (
            "GTV", "GTVp", "GTVn1", "gtv primary", "CTV_high", "CTVn",
            "PTV 60", "PTVp", "iGTV", "boost", "left lung tumor", "靶区",
        ):
            self.assertTrue(is_target(label), label)

    def test_target_subclass_returns_base_class(self) -> None:
        self.assertEqual(target_class("GTVp"), "GTV")
        self.assertEqual(target_class("CTVn_54Gy"), "CTV")
        self.assertEqual(target_class("PTVp"), "PTV")

    def test_organs_are_not_targets(self) -> None:
        for label in ("liver", "left parotid", "spinal cord", "heart"):
            self.assertFalse(is_target(label), label)

    def test_organ_names_containing_target_letters_are_not_targets(self) -> None:
        # Word-boundary matching keeps these out of the target branch.
        self.assertFalse(is_target("rectum"))
        self.assertFalse(is_target("optic chiasm"))


class CanonicalRoiTests(unittest.TestCase):
    def test_separator_and_explicit_aliases_normalize(self) -> None:
        self.assertEqual(canonical_roi(" Left-Eye "), "left eye")
        self.assertEqual(canonical_roi("Brain_Stem"), "brainstem")
        self.assertEqual(canonical_roi("Eye_L"), "left eye")
        self.assertEqual(canonical_roi("Parotid_R"), "right parotid gland")

    def test_related_names_remain_distinct(self) -> None:
        self.assertNotEqual(canonical_roi("brain"), canonical_roi("brainstem"))
        self.assertNotEqual(canonical_roi("colon"), canonical_roi("sigmoid colon"))
        self.assertNotEqual(canonical_roi("eye"), canonical_roi("left eye"))


class DiceTests(unittest.TestCase):
    def test_identical_masks_score_one(self) -> None:
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        self.assertAlmostEqual(dice(mask, mask), 1.0)

    def test_disjoint_masks_score_zero(self) -> None:
        a = np.zeros((6, 6, 6), dtype=bool)
        b = np.zeros((6, 6, 6), dtype=bool)
        a[0:2, 0:2, 0:2] = True
        b[4:6, 4:6, 4:6] = True
        self.assertAlmostEqual(dice(a, b), 0.0)

    def test_two_empty_masks_agree(self) -> None:
        empty = np.zeros((4, 4, 4), dtype=bool)
        self.assertAlmostEqual(dice(empty, empty), 1.0)

    def test_half_overlap(self) -> None:
        a = np.zeros((4, 4, 4), dtype=bool)
        b = np.zeros((4, 4, 4), dtype=bool)
        a[0:2] = True
        b[1:3] = True
        self.assertAlmostEqual(dice(a, b), 0.5)


class PhysicalGeometryTests(unittest.TestCase):
    def test_surface_distance_uses_voxel_spacing(self) -> None:
        a = np.zeros((5, 3, 3), dtype=bool)
        b = np.zeros_like(a)
        a[1, 1, 1] = True
        b[3, 1, 1] = True
        self.assertEqual(nearest_surface_distance_mm(a, b, (2.5, 1.0, 1.0)), 5.0)

    def test_centroid_is_transformed_to_world_mm(self) -> None:
        affine = np.array(
            [[2.0, 0, 0, 10.0], [0, 3.0, 0, -4.0], [0, 0, 4.0, 1.0], [0, 0, 0, 1.0]]
        )
        self.assertEqual(centroid_mm([1, 2, 3], affine), [12.0, 2.0, 13.0])


class CompletenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ContourQCEngine(
            store=SimpleNamespace(),
            backend=SimpleNamespace(),
            knowledge=HybridKnowledgeBase(KNOWLEDGE_ROOT),
        )

    @staticmethod
    def case(*labels: str, **metadata):
        contours = [
            SimpleNamespace(label=label, artifact_id=f"artifact_{index:012x}")
            for index, label in enumerate(labels)
        ]
        return SimpleNamespace(contours=contours, metadata=metadata)

    def test_site_is_not_inferred_from_roi_names(self) -> None:
        result = self.engine._completeness(self.case("prostate"), None)
        self.assertEqual(result.applicability.status, "not_assessed")
        self.assertIsNone(result.site)

    def test_completeness_uses_exact_canonical_equality(self) -> None:
        result = self.engine._completeness(
            self.case("sigmoid colon", patient_sex="female"), "pelvis"
        )
        self.assertIn("sigmoid colon", result.present)
        self.assertIn("colon", result.missing)
        self.assertNotIn("prostate", result.expected)

    def test_unknown_sex_does_not_mark_sex_specific_rois_missing(self) -> None:
        result = self.engine._completeness(self.case("bladder"), "pelvis")
        self.assertEqual(result.applicability.status, "partially_applicable")
        self.assertIn("prostate", result.applicability.unresolved)
        self.assertNotIn("prostate", result.missing)
        self.assertEqual(result.applicability.source_status, "research_seed")


class QCArtifactProvenanceTests(unittest.TestCase):
    def test_comparison_metrics_link_both_masks_and_difference_overlay(self) -> None:
        data = np.zeros((4, 4, 4), dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        volume = SimpleNamespace(
            data=data,
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
        )

        class Backend:
            name = "comparison-model"
            version = "v1"

            @staticmethod
            def segment(_image_path, _names):
                return volume, np.asarray([mask])

        uploaded = ArtifactRef(
            artifact_id="artifact_000000000001",
            case_id="case_000000000001",
            kind="contour",
            label="left kidney",
            media_type="application/gzip",
            sha256="a" * 64,
        )
        model_ref = ArtifactRef(
            artifact_id="artifact_000000000002",
            case_id=uploaded.case_id,
            kind="mask",
            label="left kidney (QC model reference)",
            media_type="application/gzip",
            sha256="b" * 64,
            metadata={"artifact_role": "qc_reference"},
        )
        overlay = ArtifactRef(
            artifact_id="artifact_000000000003",
            case_id=uploaded.case_id,
            kind="overlay",
            label="left kidney QC comparison",
            media_type="image/png",
            sha256="c" * 64,
            metadata={"artifact_role": "qc_difference_overlay"},
        )
        engine = ContourQCEngine(
            store=SimpleNamespace(), backend=Backend(), with_surface=False
        )
        engine._persist_comparison_artifacts = lambda **_kwargs: (model_ref, [overlay])

        comparisons, evidence = engine._compare_organs(
            "image.nii.gz",
            volume,
            0.001,
            [uploaded],
            {uploaded.artifact_id: mask},
            [],
            "artifact_000000000000",
        )

        self.assertEqual(comparisons[0].model_mask.artifact_id, model_ref.artifact_id)
        self.assertEqual(
            comparisons[0].difference_overlays[0].artifact_id, overlay.artifact_id
        )
        dice_evidence = next(item for item in evidence if item.metric == "dice")
        self.assertIn(uploaded.artifact_id, dice_evidence.artifact_ids)
        self.assertIn(model_ref.artifact_id, dice_evidence.artifact_ids)
        self.assertIn(overlay.artifact_id, dice_evidence.artifact_ids)


class ReportReadinessTests(unittest.TestCase):
    @staticmethod
    def case(*labels: str):
        artifacts = [
            SimpleNamespace(
                artifact_id=f"artifact_{index:012x}",
                kind="contour",
                label=label,
                metadata={},
            )
            for index, label in enumerate(labels)
        ]
        return SimpleNamespace(artifacts=artifacts)

    def test_one_arbitrary_organ_does_not_unlock_comprehensive_report(self) -> None:
        result = assess_report_readiness(self.case("left kidney"))
        self.assertFalse(result.ready)
        self.assertEqual(result.missing_inputs, ["target_contours"])

    def test_comprehensive_report_requires_organs_and_target(self) -> None:
        result = assess_report_readiness(self.case("left kidney", "GTVp"))
        self.assertTrue(result.ready)
        self.assertEqual(result.organ_count, 1)
        self.assertEqual(result.target_count, 1)

    def test_available_scope_can_report_existing_structures_only(self) -> None:
        result = assess_report_readiness(self.case("left kidney"), scope="available")
        self.assertTrue(result.ready)

    def test_qc_reference_mask_does_not_unlock_report(self) -> None:
        case = SimpleNamespace(
            artifacts=[
                SimpleNamespace(
                    artifact_id="artifact_000000000000",
                    kind="mask",
                    label="left kidney (QC model reference)",
                    metadata={"artifact_role": "qc_reference"},
                )
            ]
        )
        result = assess_report_readiness(case, scope="available")
        self.assertFalse(result.ready)

    def test_human_edit_supersedes_stale_and_repeat_masks(self) -> None:
        artifacts = [
            SimpleNamespace(
                artifact_id="artifact_000000000001",
                kind="mask",
                label="left kidney",
                metadata={"source_model": "VoxTell"},
                created_at="2026-01-01T00:00:00Z",
            ),
            SimpleNamespace(
                artifact_id="artifact_000000000002",
                kind="mask",
                label="left kidney",
                metadata={"source_model": "VoxTell"},
                created_at="2026-01-03T00:00:00Z",
            ),
            SimpleNamespace(
                artifact_id="artifact_000000000003",
                kind="mask",
                label="left kidney",
                metadata={
                    "source_model": "human_edit",
                    "derived_from": "artifact_000000000001",
                },
                created_at="2026-01-02T00:00:00Z",
            ),
            SimpleNamespace(
                artifact_id="artifact_000000000004",
                kind="contour",
                label="GTVp",
                metadata={},
                created_at="2026-01-01T00:00:00Z",
            ),
            SimpleNamespace(
                artifact_id="artifact_000000000005",
                kind="contour",
                label="GTVn",
                metadata={},
                created_at="2026-01-01T00:00:00Z",
            ),
        ]
        case = SimpleNamespace(artifacts=artifacts)
        from backend.qc import _report_candidates

        selected = _report_candidates(case)
        self.assertEqual(
            [item.artifact_id for item in selected],
            [
                "artifact_000000000003",
                "artifact_000000000004",
                "artifact_000000000005",
            ],
        )


class ReportGateTests(unittest.TestCase):
    def test_report_without_contours_offers_to_contour_first(self) -> None:
        state = {"intent": {"intent": Intent.REPORT.value, "confidence": 0.8}}
        gate = pre_plan_gate(state, (), has_contours=False)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.action, PlannerAction.ASK_USER)
        self.assertEqual(gate.gate, "report_scope")

    def test_report_with_contours_proceeds(self) -> None:
        state = {"intent": {"intent": Intent.REPORT.value, "confidence": 0.8}}
        self.assertIsNone(pre_plan_gate(state, (), has_contours=True))

    def test_report_gate_does_not_repeat(self) -> None:
        state = {
            "intent": {"intent": Intent.REPORT.value, "confidence": 0.8},
            "gates_fired": ["report_scope"],
        }
        self.assertIsNone(pre_plan_gate(state, (), has_contours=False))


if __name__ == "__main__":
    unittest.main()
