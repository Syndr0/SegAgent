from __future__ import annotations

import unittest
from pathlib import Path

from backend.knowledge import HybridKnowledgeBase


KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "knowledge"


class KnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.knowledge = HybridKnowledgeBase(KNOWLEDGE_ROOT)

    def test_protocol_alias_lookup(self) -> None:
        match = self.knowledge.lookup_protocol("prostate cancer")

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.site, "Pelvis")
        self.assertIn("bladder", match.oars)
        self.assertEqual(match.matched_by, "keyword")

    def test_unknown_protocol_abstains(self) -> None:
        self.assertIsNone(self.knowledge.lookup_protocol("made up unsupported anatomy xyz"))

    def test_guideline_readme_is_not_indexed(self) -> None:
        self.assertTrue(
            all(item["source"].casefold() != "readme.md" for item in self.knowledge.guidelines)
        )


if __name__ == "__main__":
    unittest.main()
