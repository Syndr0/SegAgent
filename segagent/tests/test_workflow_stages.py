"""The workflow, not the model, decides what gets contoured at each stage.

These cover the sequencing the spec calls for: a confirmed OAR request contours
the protocol's whole list, and the tumor workflow walks blind attempt -> confirm
-> neighbours -> enriched re-prompt in order.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dialogue import (
    STAGE_OAR_FULL,
    STAGE_TUMOR_BLIND,
    STAGE_TUMOR_NEIGHBOURS,
    STAGE_TUMOR_REFINE,
    build_tumor_prompt,
    forced_structures,
    required_workflow_decision,
)
from backend.schemas import PlannerAction

PROTOCOL = ["brainstem", "spinal cord", "left parotid gland", "larynx", "mandible"]


def observation(*measurements: dict) -> dict:
    return {
        "tool": "segment",
        "data": {"segment_result": {"measurements": list(measurements)}},
    }


def typed_observation(target_type: str, *measurements: dict) -> dict:
    return {
        "tool": "segment",
        "data": {
            "target_type": target_type,
            "segment_result": {
                "request": {"target_type": target_type},
                "measurements": list(measurements),
            },
        },
    }


def found(structure: str, centroid, volume_ml=10.0) -> dict:
    return {
        "structure": structure,
        "found": True,
        "volume_ml": volume_ml,
        "centroid_voxels": centroid,
    }


class OarStageTests(unittest.TestCase):
    def test_confirmed_oar_request_contours_the_whole_protocol_list(self) -> None:
        state = {"intent": {"intent": "oar", "site": "head and neck"}, "answers": {}}
        structures, _ = forced_structures(state, PROTOCOL, [])
        self.assertEqual(structures, PROTOCOL)

    def test_oar_list_is_not_contoured_twice(self) -> None:
        state = {
            "intent": {"intent": "oar", "site": "head and neck"},
            "answers": {"oar_contoured": True},
        }
        self.assertIsNone(forced_structures(state, PROTOCOL, []))

    def test_no_override_without_a_matched_protocol(self) -> None:
        state = {"intent": {"intent": "oar"}, "answers": {}}
        self.assertIsNone(forced_structures(state, [], []))

    def test_confirmed_site_forces_lookup_then_one_full_batch(self) -> None:
        state = {"intent": {"intent": "oar", "site": "Thorax"}, "answers": {}}
        lookup = required_workflow_decision(state, [], [], PROTOCOL)
        self.assertEqual(lookup.action, PlannerAction.LOOKUP_PROTOCOL)
        protocol_observation = {
            "tool": "lookup_protocol",
            "data": {"matched": True, "oars": PROTOCOL},
        }
        batch = required_workflow_decision(
            state, PROTOCOL, [protocol_observation], PROTOCOL
        )
        self.assertEqual(batch.action, PlannerAction.SEGMENT)
        self.assertEqual(batch.structures, PROTOCOL)


class QcStageTests(unittest.TestCase):
    def test_qc_intent_forces_qc_before_planner_prose(self) -> None:
        state = {"intent": {"intent": "qc"}, "answers": {}}
        decision = required_workflow_decision(state, [], [], PROTOCOL)
        self.assertEqual(decision.action, PlannerAction.RUN_QC)


class TumorStageTests(unittest.TestCase):
    def test_blind_attempt_uses_the_fixed_prompt(self) -> None:
        state = {"intent": {"intent": "gtv"}, "answers": {"gtv_blind_attempt": True}}
        structures, stage = forced_structures(state, PROTOCOL, [])
        self.assertEqual(structures, ["carcinoma"])
        self.assertEqual(stage, STAGE_TUMOR_BLIND)

        decision = required_workflow_decision(state, PROTOCOL, [], PROTOCOL)
        self.assertEqual(decision.action, PlannerAction.SEGMENT)
        self.assertEqual(decision.structures, ["carcinoma"])
        self.assertEqual(decision.workflow_stage.value, STAGE_TUMOR_BLIND)

    def test_spatial_target_keeps_user_description_for_initial_attempt(self) -> None:
        description = "tumor in the right upper lobe"
        state = {
            "question": f"segment {description}",
            "intent": {
                "intent": "gtv",
                "source_description": description,
            },
            "answers": {},
        }

        decision = required_workflow_decision(state, PROTOCOL, [], PROTOCOL)

        self.assertEqual(decision.action, PlannerAction.SEGMENT)
        self.assertEqual(decision.structures, [description])
        self.assertEqual(decision.workflow_stage.value, "tumor_initial")

    def test_after_confirmation_neighbouring_organs_are_contoured(self) -> None:
        state = {
            "intent": {"intent": "gtv", "targets": ["larynx"]},
            "answers": {"gtv_blind_attempt": True, "gtv_first_mask": True,
                        "gtv_location_confirmed": True},
        }
        structures, stage = forced_structures(state, PROTOCOL, [])
        self.assertEqual(structures, ["larynx"])
        self.assertEqual(stage, STAGE_TUMOR_NEIGHBOURS)

    def test_after_neighbours_the_target_is_re_prompted_spatially(self) -> None:
        observations = [
            observation(found("carcinoma", [250.0, 200.0, 60.0], 33.1)),
            observation(
                found("larynx", [250.0, 200.0, 40.0]),
                found("mandible", [250.0, 240.0, 62.0]),
            ),
        ]
        state = {
            "intent": {"intent": "gtv"},
            "answers": {"gtv_location_confirmed": True, "gtv_neighbours": True},
        }
        structures, stage = forced_structures(state, PROTOCOL, observations)
        self.assertEqual(len(structures), 1)
        self.assertEqual(stage, STAGE_TUMOR_REFINE)
        # The enriched prompt must be spatial language, not the bare word.
        self.assertNotEqual(structures[0], "tumor")
        self.assertIn("larynx", structures[0])
        self.assertRegex(structures[0], r"superior to|inferior to|anterior to|posterior to|within")

    def test_workflow_stops_dictating_once_refined(self) -> None:
        state = {
            "intent": {"intent": "gtv"},
            "answers": {"gtv_location_confirmed": True, "gtv_neighbours": True,
                        "gtv_refined": True},
        }
        self.assertIsNone(forced_structures(state, PROTOCOL, []))


class TumorPromptTests(unittest.TestCase):
    def test_refinement_uses_typed_gtv_even_when_display_label_varies(self) -> None:
        for label in (
            "multiple lesions",
            "masses",
            "carcinomas",
            "GTVp",
            "possible abnormality",
        ):
            with self.subTest(label=label):
                observations = [
                    typed_observation(
                        "gtv",
                        found(label, [250.0, 200.0, 60.0], 33.1),
                    ),
                    typed_observation(
                        "organ",
                        found("larynx", [250.0, 200.0, 40.0]),
                    ),
                ]

                prompt = build_tumor_prompt(observations)

                self.assertIsNotNone(prompt)
                self.assertIn("larynx", prompt)

    def test_prompt_names_the_nearest_organ_and_a_direction(self) -> None:
        observations = [
            observation(found("tumor", [250.0, 200.0, 60.0], 33.1)),
            observation(
                found("larynx", [250.0, 200.0, 40.0]),
                found("mandible", [250.0, 240.0, 62.0]),
            ),
        ]
        prompt = build_tumor_prompt(observations)
        # The nearest organ anchors the prompt and carries a direction, so the
        # spatial cue the prompt exists for is not thrown away.
        self.assertIn("superior to the larynx", prompt)  # tumor z=60 vs larynx z=40
        self.assertIn("posterior to the mandible", prompt)  # tumor y=200 vs 240
        self.assertIn("33.1", prompt)

    def test_target_inside_an_organ_reads_as_within_it(self) -> None:
        observations = [
            observation(found("tumor", [250.0, 200.0, 60.0], 12.0)),
            observation(found("liver", [250.0, 200.0, 60.5])),
        ]
        self.assertIn("within the liver", build_tumor_prompt(observations))

    def test_no_prompt_without_organs_to_anchor_to(self) -> None:
        self.assertIsNone(build_tumor_prompt([observation(found("tumor", [1.0, 2.0, 3.0]))]))

    def test_no_prompt_without_a_tumor(self) -> None:
        self.assertIsNone(build_tumor_prompt([observation(found("larynx", [1.0, 2.0, 3.0]))]))


if __name__ == "__main__":
    unittest.main()
