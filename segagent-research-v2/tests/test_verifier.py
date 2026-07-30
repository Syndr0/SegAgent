from __future__ import annotations

import unittest

from backend.schemas import EvidenceRecord, MetricEvidence, ToolObservation
from backend.tools import EvidenceCritic
from backend.verifier import GroundingVerifier


def seg_record(values: dict, citations=None, data=None) -> EvidenceRecord:
    return EvidenceRecord(
        record_id="evidence_seg", tool="segment",
        values=values, citations=citations or [], data=data or {},
    )


class GroundingVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = GroundingVerifier()

    def test_grounded_number_passes(self) -> None:
        rec = seg_record({"volume_ml": 142.5})
        findings = self.verifier.verify("The left kidney measures 142.5 mL.", [rec])
        self.assertEqual(findings, [])

    def test_fabricated_number_is_flagged(self) -> None:
        rec = seg_record({"volume_ml": 142.5})
        findings = self.verifier.verify("The lesion measures 37 mm.", [rec])
        self.assertTrue(any(f.check == "numeric_grounding" for f in findings))

    def test_not_fooled_by_hashes_in_data(self) -> None:
        # Regression: v2 dumped item.data (full of hex ids) and substring-matched,
        # so a fabricated "37" was "grounded" by any hash containing '37'. The new
        # verifier grounds only against emitted values, ignoring data entirely.
        rec = seg_record(
            {"volume_ml": 142.5},
            data={"mask_id": "artifact_37ab37cd", "sha256": "3773be37" * 8},
        )
        findings = self.verifier.verify("The lesion measures 37 mm.", [rec])
        self.assertTrue(any(f.check == "numeric_grounding" for f in findings))

    def test_rounding_tolerance(self) -> None:
        rec = seg_record({"dice": 0.823})
        self.assertEqual(self.verifier.verify("Dice 0.823 vs the expert.", [rec]), [])

    def test_same_number_with_wrong_metric_and_unit_is_rejected(self) -> None:
        rec = EvidenceRecord(
            record_id="evidence_qc",
            tool="run_qc",
            metrics=[
                MetricEvidence(
                    evidence_id="metric_volume",
                    structure="left kidney",
                    metric="volume",
                    value=10.0,
                    unit="mL",
                    source="uploaded contour",
                    artifact_ids=["artifact_000000000000"],
                )
            ],
        )
        hd95 = self.verifier.verify("HD95 is 10 mm.", [rec])
        count = self.verifier.verify("There are 10 structures.", [rec])
        self.assertTrue(any(f.check == "numeric_grounding" for f in hd95))
        self.assertTrue(any(f.check == "numeric_grounding" for f in count))

    def test_matching_metric_and_convertible_unit_pass(self) -> None:
        rec = EvidenceRecord(
            record_id="evidence_qc",
            tool="run_qc",
            metrics=[
                MetricEvidence(
                    evidence_id="metric_hd95",
                    metric="hd95",
                    value=10.0,
                    unit="mm",
                    source="comparison",
                )
            ],
        )
        self.assertEqual(self.verifier.verify("HD95 is 10 mm.", [rec]), [])
        self.assertEqual(self.verifier.verify("HD95 is 1 cm.", [rec]), [])

    def test_unsupported_citation_flagged(self) -> None:
        rec = seg_record({"volume_ml": 10.0}, citations=["[Local Atlas v2]"])
        ok = self.verifier.verify("Per [Local Atlas v2] the value is 10.0.", [rec])
        self.assertEqual(ok, [])
        bad = self.verifier.verify("Per [RTOG 9999] the value is 10.0.", [rec])
        self.assertTrue(any(f.check == "citation_grounding" for f in bad))

    def test_specific_claims_without_evidence_warn(self) -> None:
        findings = self.verifier.verify("The volume is 42.0 mL.", [])
        self.assertTrue(any(f.check == "evidence_presence" for f in findings))

    def test_descriptive_answer_without_evidence_is_allowed(self) -> None:
        # Visual question answering has no tool evidence by design; a purely
        # descriptive reply must not be flagged.
        findings = self.verifier.verify(
            "This is a 3D chest CT; both lungs and the mediastinum are visible.", []
        )
        self.assertEqual(findings, [])

    def test_bracketed_measurements_are_not_treated_as_citations(self) -> None:
        # Answers legitimately quote bbox arrays (the segment tool emits each
        # coordinate as a value); those brackets must not read as sources.
        rec = seg_record({
            "volume_ml": 26.606,
            "bbox0_0": 240.0, "bbox0_1": 244.0, "bbox0_2": 81.0,
            "bbox1_0": 271.0, "bbox1_1": 268.0, "bbox1_2": 100.0,
        })
        findings = self.verifier.verify(
            "The brainstem is 26.606 mL with bbox [[240, 244, 81], [271, 268, 100]].",
            [rec],
        )
        self.assertEqual([f.check for f in findings], [])

    def test_real_citation_is_still_checked(self) -> None:
        rec = seg_record({"volume_ml": 26.606})
        findings = self.verifier.verify("Per [RTOG 1016] this is fine.", [rec])
        self.assertTrue(any(f.check == "citation_grounding" for f in findings))


class EvidenceCriticQCTests(unittest.TestCase):
    def test_world_centroid_is_admitted_with_mm_units(self) -> None:
        observation = ToolObservation(
            observation_id="observation_000000000002",
            tool="inspect_mask",
            summary="Measured mask",
            data={
                "segment_result": {
                    "measurements": [
                        {
                            "structure": "left eye",
                            "found": True,
                            "voxels": 20,
                            "centroid_voxels": [3.0, 4.0, 5.0],
                            "centroid_mm": [30.0, 40.0, 50.0],
                        }
                    ]
                }
            },
        )
        critic = EvidenceCritic()
        findings = critic.review(
            "The left-eye world centroid is 30.0, 40.0, 50.0 mm.",
            [observation],
        )
        self.assertEqual(findings, [])

    def test_qc_report_metrics_and_source_are_admitted(self) -> None:
        observation = ToolObservation(
            observation_id="observation_000000000000",
            tool="run_qc",
            summary="QC",
            data={
                "qc_report": {
                    "metric_evidence": [
                        {
                            "evidence_id": "qc:left_kidney:hd95",
                            "structure": "left kidney",
                            "canonical_roi": "left kidney",
                            "metric": "hd95",
                            "value": 3.2,
                            "unit": "mm",
                            "source": "uploaded contour vs VoxTell",
                            "artifact_ids": ["artifact_000000000000"],
                        }
                    ],
                    "completeness": {
                        "citations": ["[Approved protocol v3]"],
                        "source": "Institution-approved protocol v3",
                    },
                }
            },
            evidence_ids=["artifact_000000000000"],
        )
        critic = EvidenceCritic()
        findings = critic.review(
            "Left kidney HD95 is 3.2 mm per [Approved protocol v3].",
            [observation],
        )
        self.assertEqual(findings, [])
        record = critic._record(observation)
        self.assertEqual(record.metrics[0].unit, "mm")
        self.assertIn("artifact_000000000000", record.artifact_ids)

    def test_report_bundle_metrics_are_admitted(self) -> None:
        observation = ToolObservation(
            observation_id="observation_000000000001",
            tool="build_report_evidence",
            summary="Report evidence",
            data={
                "report_evidence": {
                    "metric_evidence": [
                        {
                            "evidence_id": "report:gtvp:volume",
                            "structure": "GTVp",
                            "canonical_roi": "gtvp",
                            "metric": "volume",
                            "value": 12.4,
                            "unit": "mL",
                            "source": "uploaded target contour",
                            "artifact_ids": ["artifact_000000000001"],
                        }
                    ]
                }
            },
            evidence_ids=["artifact_000000000001"],
        )
        record = EvidenceCritic._record(observation)
        self.assertEqual(record.values["report0_volume"], 12.4)
        self.assertEqual(record.metrics[0].source, "uploaded target contour")


if __name__ == "__main__":
    unittest.main()
