from __future__ import annotations

import unittest

from backend.ledger import EvidenceLedger, new_record_id
from backend.schemas import EvidenceRecord, EvidenceStatus, TargetType


def record(record_id: str, status: EvidenceStatus = EvidenceStatus.PROPOSED, **kw) -> EvidenceRecord:
    return EvidenceRecord(record_id=record_id, tool="segment", status=status, **kw)


class EvidenceLedgerTests(unittest.TestCase):
    def test_admit_and_reject_change_admissibility(self) -> None:
        ledger = EvidenceLedger([record("evidence_a"), record("evidence_b")])
        ledger.admit("evidence_a")
        ledger.admit("evidence_b")
        ledger.reject("evidence_b")
        admitted_ids = [item.record_id for item in ledger.admitted()]
        self.assertEqual(admitted_ids, ["evidence_a"])

    def test_rejected_mask_measurements_cannot_ground_answer(self) -> None:
        # Regression for the v2 bug: a rejected segmentation's numbers stayed usable.
        ledger = EvidenceLedger()
        ledger.add(record("evidence_seg", status=EvidenceStatus.ADMITTED,
                           values={"volume_ml": 142.5}, target_type=TargetType.OAR))
        ledger.reject("evidence_seg")
        self.assertEqual(ledger.admitted(), [])

    def test_supersede_marks_old_and_links_provenance(self) -> None:
        ledger = EvidenceLedger([record("evidence_model", status=EvidenceStatus.ADMITTED)])
        edited = record("evidence_edit", status=EvidenceStatus.ADMITTED,
                        model_name="human_edit", values={"volume_ml": 99.0})
        ledger.supersede("evidence_model", edited)
        self.assertEqual(ledger.get("evidence_model").status, EvidenceStatus.SUPERSEDED)
        stored = ledger.get("evidence_edit")
        self.assertEqual(stored.status, EvidenceStatus.ADMITTED)
        self.assertEqual(stored.derived_from, "evidence_model")
        self.assertEqual([r.record_id for r in ledger.admitted()], ["evidence_edit"])

    def test_duplicate_id_rejected(self) -> None:
        ledger = EvidenceLedger([record("evidence_a")])
        with self.assertRaises(ValueError):
            ledger.add(record("evidence_a"))

    def test_new_record_id_is_prefixed_and_unique(self) -> None:
        ids = {new_record_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(i.startswith("evidence_") for i in ids))


if __name__ == "__main__":
    unittest.main()
