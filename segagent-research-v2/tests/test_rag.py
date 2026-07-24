from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.knowledge import HybridKnowledgeBase

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "knowledge"


class RagRetrievalTests(unittest.TestCase):
    def test_front_matter_is_parsed(self) -> None:
        meta, body = HybridKnowledgeBase._parse_front_matter(
            "source: Brouwer 2015\ncitation: [Brouwer 2015 § Parotid]\n\n# Parotid\nBody text."
        )
        self.assertEqual(meta["source"], "Brouwer 2015")
        self.assertEqual(meta["citation"], "[Brouwer 2015 § Parotid]")
        self.assertTrue(body.lstrip().startswith("# Parotid"))

    def test_chunks_carry_citation_and_skip_readme(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "README.md").write_text("# ignore\nnot indexed", encoding="utf-8")
            (folder / "atlas.md").write_text(
                "citation: [Test Atlas v1]\n\n# Parotid gland\n\n"
                + ("The parotid gland is bounded anteriorly by the masseter. " * 40),
                encoding="utf-8",
            )
            chunks = HybridKnowledgeBase._load_guidelines(folder)
            self.assertTrue(chunks)
            self.assertTrue(all(c["source"].casefold() != "readme.md" for c in chunks))
            self.assertTrue(all(c["citation"] == "[Test Atlas v1]" for c in chunks))
            self.assertTrue(any(c["section"] == "Parotid gland" for c in chunks))

    def test_entity_keyed_retrieval_returns_per_structure(self) -> None:
        knowledge = HybridKnowledgeBase(KNOWLEDGE_ROOT)
        result = knowledge.retrieve_for_structures(["left parotid gland", "spinal cord"])
        self.assertEqual(set(result), {"left parotid gland", "spinal cord"})
        self.assertIsInstance(result["spinal cord"], list)


if __name__ == "__main__":
    unittest.main()
