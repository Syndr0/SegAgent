from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.batching import predict_structure_chunks
from backend.planner import QwenStructuredPlanner

try:
    from backend.config import Settings
except ImportError as exc:
    Settings = None
    SETTINGS_IMPORT_ERROR = str(exc)
else:
    SETTINGS_IMPORT_ERROR = ""


@unittest.skipIf(
    Settings is None,
    f"Settings dependencies are unavailable: {SETTINGS_IMPORT_ERROR}",
)
class SettingsMemoryTests(unittest.TestCase):
    def make_settings(self, **overrides):
        # Do not let a developer's SEGAGENT_* environment or repository .env
        # change assertions about the declared defaults.
        with patch.dict(os.environ, {}, clear=True):
            return Settings(_env_file=None, **overrides)

    def test_memory_defaults_and_separate_device_fallback(self) -> None:
        defaults = self.make_settings()
        self.assertEqual(defaults.resolved_llm_device, "auto")
        self.assertEqual(defaults.resolved_voxtell_device, "auto")
        self.assertEqual(defaults.llm_device_map, "balanced_low_0")
        self.assertIsNone(defaults.llm_input_device)
        self.assertEqual(defaults.llm_max_memory, {})
        self.assertEqual(defaults.llm_max_concurrent_generations, 1)
        self.assertEqual(defaults.llm_max_prompt_images, 12)
        self.assertEqual(defaults.llm_max_artifacts, 2)
        self.assertEqual(defaults.llm_mask_axial_views, 1)
        self.assertEqual(defaults.llm_min_image_tokens, 256)
        self.assertEqual(defaults.llm_max_image_tokens, 512)
        self.assertEqual(defaults.voxtell_batch_size, 4)

        separated = self.make_settings(
            device="cuda:7",
            llm_device=" cuda:0 ",
            voxtell_device=" cuda:1 ",
        )
        self.assertEqual(separated.resolved_llm_device, "cuda:0")
        self.assertEqual(separated.resolved_voxtell_device, "cuda:1")

        fallback = self.make_settings(
            device="cuda:7",
            llm_device="cuda:0",
            voxtell_device=None,
        )
        self.assertEqual(fallback.resolved_llm_device, "cuda:0")
        self.assertEqual(fallback.resolved_voxtell_device, "cuda:7")

    def test_minimum_image_tokens_cannot_exceed_maximum(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "llm_min_image_tokens cannot exceed llm_max_image_tokens",
        ):
            self.make_settings(
                llm_min_image_tokens=513,
                llm_max_image_tokens=512,
            )

    def test_json_max_memory_environment_excludes_reserved_gpu(self) -> None:
        payload = {
            "0": "18GiB",
            "1": "18GiB",
            "2": "18GiB",
            "cpu": "64GiB",
        }
        with patch.dict(
            os.environ,
            {"SEGAGENT_LLM_MAX_MEMORY": json.dumps(payload)},
            clear=True,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.llm_max_memory, payload)
        self.assertNotIn("3", settings.llm_max_memory)

    def test_checked_in_four_gpu_profile_decodes(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(
                _env_file=(
                    Path(__file__).resolve().parents[1] / ".env.4x3090.example"
                )
            )

        self.assertEqual(settings.llm_input_device, "cuda:0")
        self.assertEqual(settings.resolved_voxtell_device, "cuda:3")
        self.assertEqual(settings.voxtell_batch_size, 4)
        self.assertNotIn("3", settings.llm_max_memory)


class PlannerMemoryHelperTests(unittest.TestCase):
    def test_max_memory_normalizes_numeric_and_cuda_device_keys(self) -> None:
        normalized = QwenStructuredPlanner._normalized_max_memory(
            {
                "0": "18GiB",
                " cuda:1 ": "17GiB",
                "cpu": "64GiB",
                "disk": "128GiB",
            }
        )

        self.assertEqual(
            normalized,
            {
                0: "18GiB",
                1: "17GiB",
                "cpu": "64GiB",
                "disk": "128GiB",
            },
        )

    def test_artifact_limit_keeps_named_mask_ahead_of_many_visible_masks(self) -> None:
        ambient = [
            SimpleNamespace(
                artifact_id=f"artifact_{index}",
                label=f"organ {index}",
                kind="mask",
                metadata={},
            )
            for index in range(20)
        ]
        named = SimpleNamespace(
            artifact_id="artifact_left_eye",
            label="left eye",
            kind="mask",
            metadata={},
        )
        case = SimpleNamespace(artifacts=[*ambient, named])

        selected = QwenStructuredPlanner._artifact_view_ids(
            case,
            {
                "intent": {"targets": ["left eye"]},
                "message_context": {
                    "visible_mask_ids": [
                        artifact.artifact_id for artifact in [*ambient, named]
                    ],
                },
            },
            limit=1,
        )

        self.assertEqual(selected, ["artifact_left_eye"])
        self.assertLessEqual(len(selected), 1)

    def test_artifact_views_respect_aggregate_prompt_image_budget(self) -> None:
        artifacts = [
            SimpleNamespace(
                artifact_id=f"artifact_{index}",
                label=f"organ {index}",
                kind="mask",
                metadata={},
            )
            for index in range(2)
        ]
        store = SimpleNamespace(
            artifact_path=lambda artifact: artifact.artifact_id,
        )
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=store,
            max_prompt_images=6,
            max_artifacts=2,
            mask_axial_views=1,
        )
        fake_imaging = ModuleType("backend.imaging")
        fake_imaging.load_mask = lambda path, volume: path
        fake_imaging.mask_grounding_views = lambda volume, mask, count: [
            (f"view {index}", object()) for index in range(3)
        ]
        content = [{"type": "image", "image": object()} for _ in range(5)]

        with patch.dict(sys.modules, {"backend.imaging": fake_imaging}):
            planner._append_artifact_views(
                content,
                SimpleNamespace(artifacts=artifacts),
                SimpleNamespace(data=object()),
                {
                    "message_context": {
                        "referenced_artifact_ids": [
                            artifact.artifact_id for artifact in artifacts
                        ],
                    }
                },
            )

        self.assertEqual(
            sum(item.get("type") == "image" for item in content),
            6,
        )

    def test_qwen35_image_budget_uses_checkpoint_patch_geometry(self) -> None:
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=SimpleNamespace(),
            min_image_tokens=256,
            max_image_tokens=512,
        )
        planner.image_patch_size = 16
        planner.image_merge_size = 2

        item = planner._image_item(object())

        self.assertEqual(item["min_pixels"], 256 * 32 * 32)
        self.assertEqual(item["max_pixels"], 512 * 32 * 32)

    def test_balanced_low_zero_places_generation_outputs_on_gpu_zero(self) -> None:
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=SimpleNamespace(),
            device_map="balanced_low_0",
            max_memory={"0": "18GiB", "1": "18GiB", "2": "18GiB"},
        )
        self.assertEqual(
            planner._select_input_device("cuda", "cuda:1"),
            "cuda:0",
        )

        planner.max_memory = {"1": "18GiB", "2": "18GiB"}
        self.assertEqual(
            planner._select_input_device("cuda", "cuda:1"),
            "cuda:1",
        )

    def test_generation_semaphore_serializes_model_calls(self) -> None:
        planner = QwenStructuredPlanner(
            model_id="unused",
            store=SimpleNamespace(),
            max_concurrent_generations=1,
        )
        start = Barrier(3)
        concurrent_entry = Event()
        state_lock = Lock()
        active = 0
        peak_active = 0
        entry_count = 0
        results: list[str] = []
        failures: list[BaseException] = []

        def fake_generate_once(*args, **kwargs) -> str:
            nonlocal active, peak_active, entry_count
            del args, kwargs
            with state_lock:
                active += 1
                entry_count += 1
                ordinal = entry_count
                peak_active = max(peak_active, active)
                if active > 1:
                    concurrent_entry.set()
            try:
                if ordinal == 1:
                    # Without the semaphore, the second worker enters during
                    # this short window and raises peak_active to two.
                    concurrent_entry.wait(timeout=0.1)
                return "generated"
            finally:
                with state_lock:
                    active -= 1

        def invoke() -> None:
            try:
                start.wait(timeout=2)
                results.append(planner._generate([]))
            except BaseException as exc:
                failures.append(exc)

        with patch.object(planner, "_generate_once", side_effect=fake_generate_once):
            workers = [Thread(target=invoke) for _ in range(2)]
            for worker in workers:
                worker.start()
            start.wait(timeout=2)
            for worker in workers:
                worker.join(timeout=2)

        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(failures, [])
        self.assertEqual(results, ["generated", "generated"])
        self.assertEqual(peak_active, 1)


class VoxTellBatchingTests(unittest.TestCase):
    def test_large_structure_set_is_chunked_without_reordering_masks(self) -> None:
        structures = [f"organ {index}" for index in range(5)]
        values = {name: index + 1 for index, name in enumerate(structures)}
        calls: list[list[str]] = []

        def predict(image, prompts):
            self.assertEqual(image, "image")
            calls.append(list(prompts))
            masks = np.stack(
                [
                    np.full((2, 3, 4), values[prompt], dtype=np.uint8)
                    for prompt in prompts
                ]
            )
            return masks[0] if len(prompts) == 1 else masks

        masks = predict_structure_chunks(
            predict,
            "image",
            structures,
            batch_size=2,
        )

        self.assertEqual(
            calls,
            [["organ 0", "organ 1"], ["organ 2", "organ 3"], ["organ 4"]],
        )
        self.assertEqual(masks.shape, (5, 2, 3, 4))
        self.assertEqual([int(mask[0, 0, 0]) for mask in masks], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
