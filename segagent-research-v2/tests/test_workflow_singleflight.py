from __future__ import annotations

import io
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.storage import ResearchStore
try:
    from backend.workflow import (
        ConversationConflictError,
        WorkflowService,
        event,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("langgraph"):
        raise unittest.SkipTest(f"workflow dependency unavailable: {exc}") from exc
    raise


class FakeGraphRuntime:
    def __init__(self):
        self.mode = "complete"
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def stream(self, value, *, config, stream_mode):
        del value, config, stream_mode
        if self.mode == "fail":
            raise RuntimeError("synthetic graph failure")
        if self.mode == "question":
            yield (
                "updates",
                {
                    "__interrupt__": [
                        SimpleNamespace(
                            value={
                                "question": "Which treatment site?",
                                "gate": "site",
                            }
                        )
                    ]
                },
            )
            return
        if self.mode == "nonterminal":
            yield (
                "custom",
                event("phase_started", phase="synthetic"),
            )
            return
        if self.mode == "blocking":
            self.entered.set()
            if not self.unblock.wait(timeout=5):
                raise RuntimeError("blocking test timed out")
            yield (
                "custom",
                event("phase_started", phase="unblocked"),
            )
            return
        yield (
            "updates",
            {"finish": {"events": [event("run_completed")]}},
        )


class WorkflowSingleFlightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ResearchStore(Path(self.temp.name))
        self.case = self.store.create_case(
            "scan.nii.gz",
            io.BytesIO(b"not-a-real-nifti"),
        )
        self.runtime = FakeGraphRuntime()
        self.service = WorkflowService(
            self.store,
            SimpleNamespace(graph=self.runtime),
        )

    def test_overlapping_send_for_same_conversation_is_rejected(self) -> None:
        run_id, first = self.service.send(self.case.case_id, "first")
        with self.assertRaises(ConversationConflictError):
            self.service.send(self.case.case_id, "second")

        first.close()
        self.assertEqual(self.store.get_run(run_id).status, "failed")
        _, after_close = self.service.send(self.case.case_id, "after close")
        after_close.close()

    def test_different_cases_can_stream_concurrently(self) -> None:
        other = self.store.create_case("other.nii.gz", io.BytesIO(b"other"))
        _, first = self.service.send(self.case.case_id, "first")
        _, second = self.service.send(other.case_id, "second")
        first.close()
        second.close()

    def test_guard_releases_after_stream_completion(self) -> None:
        run_id, first = self.service.send(self.case.case_id, "first")
        self.assertEqual([item.type for item in first], ["run_started", "run_completed"])
        self.assertEqual(self.store.get_run(run_id).status, "completed")

        _, second = self.service.send(self.case.case_id, "second")
        second.close()

    def test_guard_releases_and_marks_failed_after_propagated_error(self) -> None:
        original_execute = self.service._execute

        def fail_before_graph_events(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("synthetic stream failure")
            yield  # pragma: no cover - keeps this an iterator

        self.service._execute = fail_before_graph_events
        run_id, failed = self.service.send(self.case.case_id, "fail")
        self.assertEqual(next(failed).type, "run_started")
        with self.assertRaisesRegex(RuntimeError, "synthetic stream failure"):
            next(failed)
        self.assertEqual(self.store.get_run(run_id).status, "failed")

        self.service._execute = original_execute
        _, recovered = self.service.send(self.case.case_id, "recover")
        recovered.close()

    def test_graph_error_event_releases_and_marks_failed(self) -> None:
        self.runtime.mode = "fail"
        run_id, failed = self.service.send(self.case.case_id, "fail")
        self.assertEqual([item.type for item in failed], ["run_started", "error"])
        self.assertEqual(self.store.get_run(run_id).status, "failed")

        self.runtime.mode = "complete"
        _, recovered = self.service.send(self.case.case_id, "recover")
        recovered.close()

    def test_waiting_reply_continues_without_double_acquiring_guard(self) -> None:
        conversation = self.store.conversation_for_case(self.case.case_id)
        run = self.store.create_run(
            self.case.case_id,
            "Which treatment site?",
            conversation.conversation_id,
        )
        run.status = "waiting_reply"
        self.store.save_run(run)
        conversation.pending_run_id = run.run_id
        conversation.pending_question = "Which treatment site?"
        self.store.save_conversation(conversation)

        resumed_run_id, resumed = self.service.send(self.case.case_id, "head and neck")
        self.assertEqual(resumed_run_id, run.run_id)
        self.assertEqual([item.type for item in resumed], ["run_completed"])
        self.assertEqual(self.store.get_run(run.run_id).status, "completed")
        self.assertIsNone(
            self.store.conversation_for_case(self.case.case_id).pending_run_id
        )

        _, next_turn = self.service.send(self.case.case_id, "next question")
        next_turn.close()

    def test_question_event_releases_before_eof_and_old_close_is_token_safe(self) -> None:
        self.runtime.mode = "question"
        run_id, original = self.service.send(self.case.case_id, "start")
        self.assertEqual(next(original).type, "run_started")
        self.assertEqual(next(original).type, "question_pending")
        self.assertEqual(self.store.get_run(run_id).status, "waiting_reply")

        self.runtime.mode = "complete"
        resumed_run_id, resumed = self.service.send(
            self.case.case_id,
            "head and neck",
        )
        self.assertEqual(resumed_run_id, run_id)

        # Closing the old suspended stream must not discard the reply's newer
        # token and accidentally admit a third overlapping request.
        original.close()
        with self.assertRaises(ConversationConflictError):
            self.service.send(self.case.case_id, "overlap")

        self.assertEqual([item.type for item in resumed], ["run_completed"])
        self.assertEqual(self.store.get_run(run_id).status, "completed")

    def test_unstarted_and_started_abort_mark_runs_failed(self) -> None:
        unstarted_id, unstarted = self.service.send(self.case.case_id, "unstarted")
        unstarted.close()
        self.assertEqual(self.store.get_run(unstarted_id).status, "failed")

        started_id, started = self.service.send(self.case.case_id, "started")
        self.assertEqual(next(started).type, "run_started")
        started.close()
        self.assertEqual(self.store.get_run(started_id).status, "failed")

        _, recovered = self.service.send(self.case.case_id, "recovered")
        recovered.close()

    def test_unstarted_reply_has_no_message_side_effect_and_clears_pending_on_abort(
        self,
    ) -> None:
        conversation = self.store.conversation_for_case(self.case.case_id)
        run = self.store.create_run(
            self.case.case_id,
            "Which treatment site?",
            conversation.conversation_id,
        )
        run.status = "waiting_reply"
        self.store.save_run(run)
        conversation.pending_run_id = run.run_id
        conversation.pending_question = "Which treatment site?"
        self.store.save_conversation(conversation)
        messages_before = len(self.store.messages(conversation.conversation_id))

        resumed_run_id, resumed = self.service.send(self.case.case_id, "head and neck")
        self.assertEqual(resumed_run_id, run.run_id)
        self.assertEqual(
            len(self.store.messages(conversation.conversation_id)),
            messages_before,
        )
        resumed.close()

        self.assertEqual(self.store.get_run(run.run_id).status, "failed")
        reloaded = self.store.conversation_for_case(self.case.case_id)
        self.assertIsNone(reloaded.pending_run_id)
        self.assertEqual(
            len(self.store.messages(conversation.conversation_id)),
            messages_before,
        )

    def test_concurrent_close_defers_release_until_active_next_returns(self) -> None:
        self.runtime.mode = "blocking"
        run_id, stream = self.service.send(self.case.case_id, "block")
        self.assertEqual(next(stream).type, "run_started")
        outcomes: list[str] = []

        def advance() -> None:
            try:
                next(stream)
                outcomes.append("event")
            except StopIteration:
                outcomes.append("stopped")
            except BaseException as exc:  # pragma: no cover - diagnostic branch
                outcomes.append(type(exc).__name__)

        worker = threading.Thread(target=advance)
        worker.start()
        self.assertTrue(self.runtime.entered.wait(timeout=2))

        stream.close()
        with self.assertRaises(ConversationConflictError):
            self.service.send(self.case.case_id, "too early")

        self.runtime.unblock.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcomes, ["stopped"])
        self.assertEqual(self.store.get_run(run_id).status, "failed")

        _, recovered = self.service.send(self.case.case_id, "after worker")
        recovered.close()

    def test_nonterminal_exhaustion_marks_run_failed_and_releases(self) -> None:
        self.runtime.mode = "nonterminal"
        run_id, stream = self.service.send(self.case.case_id, "incomplete")
        self.assertEqual(
            [item.type for item in stream],
            ["run_started", "phase_started"],
        )
        self.assertEqual(self.store.get_run(run_id).status, "failed")

        self.runtime.mode = "complete"
        _, recovered = self.service.send(self.case.case_id, "recover")
        recovered.close()


if __name__ == "__main__":
    unittest.main()
