from __future__ import annotations

import unittest

from backend.guards import SeenRequests, request_signature


class GuardTests(unittest.TestCase):
    def test_signature_is_order_case_whitespace_insensitive(self) -> None:
        a = request_signature("case_1", "segment", [" Left Kidney ", "liver"])
        b = request_signature("case_1", "segment", ["liver", "left   kidney"])
        self.assertEqual(a, b)

    def test_signature_distinguishes_case_and_tool(self) -> None:
        base = request_signature("case_1", "segment", ["liver"])
        self.assertNotEqual(base, request_signature("case_2", "segment", ["liver"]))
        self.assertNotEqual(base, request_signature("case_1", "run_qc", ["liver"]))
        self.assertNotEqual(base, request_signature("case_1", "segment", ["spleen"]))

    def test_identical_rerun_is_blocked(self) -> None:
        seen = SeenRequests()
        sig = request_signature("case_1", "segment", ["liver"])
        self.assertTrue(seen.is_new(sig))
        self.assertFalse(seen.is_new(sig))
        self.assertTrue(seen.seen(sig))


if __name__ == "__main__":
    unittest.main()
