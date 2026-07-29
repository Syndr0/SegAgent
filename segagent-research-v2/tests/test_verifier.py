from __future__ import annotations

import unittest

from backend.schemas import EvidenceRecord
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
            "This is a chest CT; both lungs and the mediastinum are visible.", []
        )
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
