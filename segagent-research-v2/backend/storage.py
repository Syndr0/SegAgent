from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path
from typing import BinaryIO, Iterable

from .schemas import (
    ArtifactRef,
    CaseRecord,
    ChatMessage,
    ChatRole,
    ConversationRecord,
    RunEvent,
    RunRecord,
    utc_now,
)


ID_RE = re.compile(r"^[a-z]+_[0-9a-f]{12,32}$")


def safe_filename(name: str, default: str = "upload.bin") -> str:
    base = Path(name or default).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned[:180] or default


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResearchStore:
    """Case-scoped file store with append-only run events.

    This deliberately small store is suitable for reproducible local research.
    The interfaces can later be implemented with Postgres and object storage.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.cases_dir = self.root / "cases"
        self.runs_dir = self.root / "runs"
        self.conversations_dir = self.root / "conversations"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _validate_id(value: str, prefix: str) -> str:
        if not ID_RE.fullmatch(value) or not value.startswith(prefix + "_"):
            raise ValueError(f"invalid {prefix} id")
        return value

    def case_dir(self, case_id: str) -> Path:
        self._validate_id(case_id, "case")
        return self.cases_dir / case_id

    def run_path(self, run_id: str) -> Path:
        self._validate_id(run_id, "run")
        return self.runs_dir / f"{run_id}.json"

    def create_case(
        self,
        source_name: str,
        image_stream: BinaryIO,
        metadata: dict | None = None,
    ) -> CaseRecord:
        case_id = _id("case")
        folder = self.cases_dir / case_id
        artifacts = folder / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=False)
        filename = safe_filename(source_name, "image.nii.gz")
        image_path = artifacts / f"image_{filename}"
        with image_path.open("wb") as handle:
            while chunk := image_stream.read(1024 * 1024):
                handle.write(chunk)
        image_ref = ArtifactRef(
            artifact_id=_id("artifact"),
            case_id=case_id,
            kind="image",
            label=filename,
            media_type="application/gzip" if filename.endswith(".gz") else "application/octet-stream",
            sha256=_sha256(image_path),
            metadata={"relative_path": str(image_path.relative_to(folder))},
        )
        record = CaseRecord(
            case_id=case_id,
            source_name=filename,
            image=image_ref,
            artifacts=[image_ref],
            metadata=metadata or {},
        )
        self._write_case(record)
        return record

    def _write_case(self, record: CaseRecord) -> None:
        folder = self.case_dir(record.case_id)
        temp = folder / "case.json.tmp"
        temp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(folder / "case.json")

    def get_case(self, case_id: str) -> CaseRecord:
        path = self.case_dir(case_id) / "case.json"
        if not path.exists():
            raise FileNotFoundError(f"case not found: {case_id}")
        return CaseRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def artifact_path(self, ref: ArtifactRef) -> Path:
        case = self.get_case(ref.case_id)
        known = {item.artifact_id: item for item in case.artifacts}
        stored = known.get(ref.artifact_id)
        if stored is None:
            raise FileNotFoundError(f"artifact not registered: {ref.artifact_id}")
        relative = stored.metadata.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("artifact has no relative path")
        path = (self.case_dir(ref.case_id) / relative).resolve()
        case_root = self.case_dir(ref.case_id).resolve()
        if case_root not in path.parents:
            raise ValueError("artifact path escaped its case")
        return path

    def get_artifact(self, case_id: str, artifact_id: str) -> tuple[ArtifactRef, Path]:
        case = self.get_case(case_id)
        ref = next((item for item in case.artifacts if item.artifact_id == artifact_id), None)
        if ref is None:
            raise FileNotFoundError("artifact not found")
        return ref, self.artifact_path(ref)

    def artifact_inventory(self, case_id: str) -> list[dict]:
        """Return a deterministic, case-owned mask/contour registry.

        ``active`` means no newer artifact declares itself derived from this
        artifact. ``latest`` additionally identifies the most recently
        registered active artifact for the same exact label and role.
        """

        case = self.get_case(case_id)
        artifacts = [
            item for item in case.artifacts if item.kind in {"mask", "contour", "overlay"}
        ]
        superseded = {
            str(item.metadata.get("derived_from"))
            for item in artifacts
            if item.metadata.get("derived_from")
        }
        latest_by_key: dict[tuple[str, str], str] = {}
        for item in artifacts:
            if item.artifact_id in superseded:
                continue
            role = str(item.metadata.get("target_type") or item.kind)
            latest_by_key[(item.label.casefold(), role)] = item.artifact_id
        return [
            {
                "artifact_id": item.artifact_id,
                "label": item.label,
                "kind": item.kind,
                "role": str(item.metadata.get("target_type") or item.kind),
                "source_model": item.metadata.get("source_model"),
                "model_version": item.metadata.get("source_version"),
                "artifact_version": item.metadata.get("artifact_version")
                or item.metadata.get("source_version")
                or "1",
                "derived_from": item.metadata.get("derived_from"),
                "active": item.artifact_id not in superseded,
                "latest": (
                    latest_by_key.get(
                        (
                            item.label.casefold(),
                            str(item.metadata.get("target_type") or item.kind),
                        )
                    )
                    == item.artifact_id
                ),
            }
            for item in artifacts
        ]

    def allocate_artifact_path(self, case_id: str, suffix: str) -> tuple[str, Path]:
        artifact_id = _id("artifact")
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = self.case_dir(case_id) / "artifacts" / f"{artifact_id}{suffix}"
        return artifact_id, path

    def register_artifact(
        self,
        case_id: str,
        artifact_id: str,
        path: Path,
        kind: str,
        label: str,
        media_type: str,
        metadata: dict | None = None,
        contour: bool = False,
    ) -> ArtifactRef:
        path = Path(path).resolve()
        folder = self.case_dir(case_id).resolve()
        if folder not in path.parents or not path.is_file():
            raise ValueError("artifact must be an existing file inside the case")
        payload = dict(metadata or {})
        payload["relative_path"] = str(path.relative_to(folder))
        ref = ArtifactRef(
            artifact_id=artifact_id,
            case_id=case_id,
            kind=kind,
            label=label,
            media_type=media_type,
            sha256=_sha256(path),
            metadata=payload,
        )
        with self._lock:
            case = self.get_case(case_id)
            case.artifacts.append(ref)
            if contour:
                case.contours.append(ref)
            self._write_case(case)
        return ref

    def add_contour(self, case_id: str, name: str, stream: BinaryIO) -> ArtifactRef:
        return self.add_contours(case_id, [(name, stream)])[0]

    def add_contours(
        self,
        case_id: str,
        uploads: list[tuple[str, BinaryIO]],
    ) -> list[ArtifactRef]:
        """Store a contour batch and register it with one case update.

        Files are removed if any item fails, so a multi-file upload never leaves
        a partly registered batch behind.
        """
        self.get_case(case_id)
        if not uploads:
            return []
        staged: list[tuple[ArtifactRef, Path]] = []
        created_paths: list[Path] = []
        try:
            for name, stream in uploads:
                clean = safe_filename(name, "contour.nii.gz")
                suffix = ".nii.gz" if clean.casefold().endswith(".nii.gz") else ".nii"
                artifact_id, path = self.allocate_artifact_path(case_id, suffix)
                # Track the path before writing so a failed stream is cleaned up.
                created_paths.append(path)
                label = (
                    clean.removesuffix(".nii.gz")
                    .removesuffix(".nii")
                    .replace("_", " ")
                )
                with path.open("wb") as handle:
                    while chunk := stream.read(1024 * 1024):
                        handle.write(chunk)
                if path.stat().st_size == 0:
                    raise ValueError(f"contour file is empty: {clean}")
                ref = ArtifactRef(
                    artifact_id=artifact_id,
                    case_id=case_id,
                    kind="contour",
                    label=label,
                    media_type=(
                        "application/gzip"
                        if suffix == ".nii.gz"
                        else "application/octet-stream"
                    ),
                    sha256=_sha256(path),
                    metadata={
                        "relative_path": str(path.relative_to(self.case_dir(case_id)))
                    },
                )
                staged.append((ref, path))

            with self._lock:
                case = self.get_case(case_id)
                refs = [ref for ref, _ in staged]
                case.artifacts.extend(refs)
                case.contours.extend(refs)
                self._write_case(case)
            return refs
        except Exception:
            for path in created_paths:
                path.unlink(missing_ok=True)
            raise

    def conversation_path(self, conversation_id: str) -> Path:
        self._validate_id(conversation_id, "conversation")
        return self.conversations_dir / f"{conversation_id}.json"

    def save_conversation(self, conversation: ConversationRecord) -> None:
        conversation.updated_at = utc_now()
        path = self.conversation_path(conversation.conversation_id)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(conversation.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(path)

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        path = self.conversation_path(conversation_id)
        if not path.exists():
            raise FileNotFoundError("conversation not found")
        return ConversationRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def conversation_for_case(self, case_id: str) -> ConversationRecord:
        """Return the case's conversation, creating it on first use.

        One conversation per case: the id doubles as the LangGraph thread id, so
        every turn for a patient shares accumulated context.
        """
        self.get_case(case_id)
        pointer = self.case_dir(case_id) / "conversation.txt"
        with self._lock:
            if pointer.exists():
                conversation_id = pointer.read_text(encoding="utf-8").strip()
                if conversation_id:
                    try:
                        return self.get_conversation(conversation_id)
                    except (FileNotFoundError, ValueError):
                        pass
            conversation = ConversationRecord(
                conversation_id=_id("conversation"), case_id=case_id
            )
            self.save_conversation(conversation)
            pointer.write_text(conversation.conversation_id, encoding="utf-8")
            return conversation

    def append_message(
        self,
        conversation_id: str,
        case_id: str,
        role: ChatRole | str,
        text: str,
        run_id: str | None = None,
        metadata: dict | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            message_id=_id("message"),
            conversation_id=conversation_id,
            case_id=case_id,
            role=ChatRole(role),
            text=text,
            run_id=run_id,
            metadata=metadata or {},
        )
        path = self.conversations_dir / f"{conversation_id}.messages.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(message.model_dump_json() + "\n")
        return message

    def messages(self, conversation_id: str) -> list[ChatMessage]:
        self._validate_id(conversation_id, "conversation")
        path = self.conversations_dir / f"{conversation_id}.messages.jsonl"
        if not path.exists():
            return []
        return [
            ChatMessage.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def create_run(
        self, case_id: str, question: str, conversation_id: str | None = None
    ) -> RunRecord:
        self.get_case(case_id)
        run = RunRecord(
            run_id=_id("run"),
            case_id=case_id,
            question=question,
            status="created",
            conversation_id=conversation_id,
        )
        self.save_run(run)
        return run

    def save_run(self, run: RunRecord) -> None:
        run.updated_at = utc_now()
        temp = self.run_path(run.run_id).with_suffix(".json.tmp")
        temp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        temp.replace(self.run_path(run.run_id))

    def get_run(self, run_id: str) -> RunRecord:
        path = self.run_path(run_id)
        if not path.exists():
            raise FileNotFoundError("run not found")
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def append_event(self, event: RunEvent) -> None:
        run = self.get_run(event.run_id)
        if run.case_id != event.case_id:
            raise ValueError("event case does not match run")
        events_path = self.runs_dir / f"{event.run_id}.events.jsonl"
        with self._lock, events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
            run.event_count = max(run.event_count, event.sequence)
            if event.type == "answer":
                run.final_answer = str(event.payload.get("text", ""))
            self.save_run(run)

    def events(self, run_id: str) -> Iterable[RunEvent]:
        self.get_run(run_id)
        path = self.runs_dir / f"{run_id}.events.jsonl"
        if not path.exists():
            return []
        return [
            RunEvent.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
