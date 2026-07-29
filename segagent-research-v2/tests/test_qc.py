import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dialogue import pre_plan_gate
from backend.qc import dice, is_target
from backend.schemas import Intent, PlannerAction


class TargetClassificationTests(unittest.TestCase):
    def test_target_volumes_are_recognized(self) -> None:
        for label in ("GTV", "gtv primary", "CTV_high", "PTV 60", "boost", "靶区"):
            self.assertTrue(is_target(label), label)

    def test_organs_are_not_targets(self) -> None:
        for label in ("liver", "left parotid", "spinal cord", "heart"):
            self.assertFalse(is_target(label), label)

    def test_organ_names_containing_target_letters_are_not_targets(self) -> None:
        # Word-boundary matching keeps these out of the target branch.
        self.assertFalse(is_target("rectum"))
        self.assertFalse(is_target("optic chiasm"))


class DiceTests(unittest.TestCase):
    def test_identical_masks_score_one(self) -> None:
        mask = np.zeros((6, 6, 6), dtype=bool)
        mask[2:4, 2:4, 2:4] = True
        self.assertAlmostEqual(dice(mask, mask), 1.0)

    def test_disjoint_masks_score_zero(self) -> None:
        a = np.zeros((6, 6, 6), dtype=bool)
        b = np.zeros((6, 6, 6), dtype=bool)
        a[0:2, 0:2, 0:2] = True
        b[4:6, 4:6, 4:6] = True
        self.assertAlmostEqual(dice(a, b), 0.0)

    def test_two_empty_masks_agree(self) -> None:
        empty = np.zeros((4, 4, 4), dtype=bool)
        self.assertAlmostEqual(dice(empty, empty), 1.0)

    def test_half_overlap(self) -> None:
        a = np.zeros((4, 4, 4), dtype=bool)
        b = np.zeros((4, 4, 4), dtype=bool)
        a[0:2] = True
        b[1:3] = True
        self.assertAlmostEqual(dice(a, b), 0.5)


class ReportGateTests(unittest.TestCase):
    def test_report_without_contours_offers_to_contour_first(self) -> None:
        state = {"intent": {"intent": Intent.REPORT.value, "confidence": 0.8}}
        gate = pre_plan_gate(state, (), has_contours=False)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.action, PlannerAction.ASK_USER)
        self.assertEqual(gate.gate, "report_scope")

    def test_report_with_contours_proceeds(self) -> None:
        state = {"intent": {"intent": Intent.REPORT.value, "confidence": 0.8}}
        self.assertIsNone(pre_plan_gate(state, (), has_contours=True))

    def test_report_gate_does_not_repeat(self) -> None:
        state = {
            "intent": {"intent": Intent.REPORT.value, "confidence": 0.8},
            "gates_fired": ["report_scope"],
        }
        self.assertIsNone(pre_plan_gate(state, (), has_contours=False))


if __name__ == "__main__":
    unittest.main()
