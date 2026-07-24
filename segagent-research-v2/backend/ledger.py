from __future__ import annotations

import uuid

from .schemas import EvidenceRecord, EvidenceStatus


def new_record_id() -> str:
    return f"evidence_{uuid.uuid4().hex[:16]}"


class EvidenceLedger:
    """Ordered store of typed evidence with an admission lifecycle.

    The composer and verifier read only ``admitted`` records, so rejecting or
    superseding a record mechanically removes it from the grounding set. This
    is what makes "a rejected mask cannot ground the final answer" a property
    of the data model rather than of planner discipline.
    """

    def __init__(self, records: list[EvidenceRecord] | None = None):
        self._records: dict[str, EvidenceRecord] = {}
        self._order: list[str] = []
        for record in records or []:
            self.add(record)

    def add(self, record: EvidenceRecord) -> EvidenceRecord:
        if record.record_id in self._records:
            raise ValueError(f"duplicate evidence id: {record.record_id}")
        self._records[record.record_id] = record
        self._order.append(record.record_id)
        return record

    def get(self, record_id: str) -> EvidenceRecord:
        return self._records[record_id]

    def _update_status(self, record_id: str, status: EvidenceStatus) -> EvidenceRecord:
        if record_id not in self._records:
            raise KeyError(record_id)
        updated = self._records[record_id].model_copy(update={"status": status})
        self._records[record_id] = updated
        return updated

    def admit(self, record_id: str) -> EvidenceRecord:
        return self._update_status(record_id, EvidenceStatus.ADMITTED)

    def reject(self, record_id: str) -> EvidenceRecord:
        return self._update_status(record_id, EvidenceStatus.REJECTED)

    def supersede(self, old_id: str, new_record: EvidenceRecord) -> EvidenceRecord:
        """Mark ``old_id`` superseded and add ``new_record`` in its place.

        Used when a human edit replaces a model mask: the model's measurements
        stop being admissible and the edited record carries the provenance.
        """
        self._update_status(old_id, EvidenceStatus.SUPERSEDED)
        if new_record.derived_from is None:
            new_record = new_record.model_copy(update={"derived_from": old_id})
        return self.add(new_record)

    def records(self) -> list[EvidenceRecord]:
        return [self._records[record_id] for record_id in self._order]

    def by_status(self, status: EvidenceStatus) -> list[EvidenceRecord]:
        return [record for record in self.records() if record.status == status]

    def admitted(self) -> list[EvidenceRecord]:
        return self.by_status(EvidenceStatus.ADMITTED)
