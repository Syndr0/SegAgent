import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.schemas import ChatRole
from backend.storage import ResearchStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ResearchStore(Path(self.temp.name))
        self.case = self.store.create_case("scan.nii.gz", io.BytesIO(b"not-a-real-nifti"))

    def test_conversation_is_created_once_per_case(self) -> None:
        first = self.store.conversation_for_case(self.case.case_id)
        second = self.store.conversation_for_case(self.case.case_id)
        self.assertEqual(first.conversation_id, second.conversation_id)

    def test_messages_round_trip_in_order(self) -> None:
        conversation = self.store.conversation_for_case(self.case.case_id)
        self.store.append_message(
            conversation.conversation_id, self.case.case_id, ChatRole.USER, "segment the liver"
        )
        self.store.append_message(
            conversation.conversation_id, self.case.case_id, ChatRole.AGENT, "done"
        )
        messages = self.store.messages(conversation.conversation_id)
        self.assertEqual([item.text for item in messages], ["segment the liver", "done"])
        self.assertEqual(
            [item.role for item in messages], [ChatRole.USER, ChatRole.AGENT]
        )

    def test_pending_question_survives_reload(self) -> None:
        conversation = self.store.conversation_for_case(self.case.case_id)
        run = self.store.create_run(
            self.case.case_id, "find the OAR", conversation.conversation_id
        )
        conversation.pending_question = "Which treatment site?"
        conversation.pending_run_id = run.run_id
        self.store.save_conversation(conversation)

        reloaded = self.store.conversation_for_case(self.case.case_id)
        self.assertEqual(reloaded.pending_question, "Which treatment site?")
        self.assertEqual(reloaded.pending_run_id, run.run_id)

    def test_run_records_its_conversation(self) -> None:
        conversation = self.store.conversation_for_case(self.case.case_id)
        run = self.store.create_run(
            self.case.case_id, "segment the liver", conversation.conversation_id
        )
        self.assertEqual(
            self.store.get_run(run.run_id).conversation_id, conversation.conversation_id
        )

    def test_messages_are_isolated_between_cases(self) -> None:
        other = self.store.create_case("other.nii.gz", io.BytesIO(b"other"))
        mine = self.store.conversation_for_case(self.case.case_id)
        theirs = self.store.conversation_for_case(other.case_id)
        self.store.append_message(
            mine.conversation_id, self.case.case_id, ChatRole.USER, "mine"
        )
        self.assertEqual([m.text for m in self.store.messages(mine.conversation_id)], ["mine"])
        self.assertEqual(self.store.messages(theirs.conversation_id), [])


if __name__ == "__main__":
    unittest.main()
