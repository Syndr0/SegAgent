from __future__ import annotations

import unittest
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.prompts import trusted_prompt_normalizations
from backend.schemas import (
    ArtifactRef,
    PromptCategory,
    SegmentRequest,
    TargetType,
)
from backend.storage import ResearchStore
from backend.tools import MaskInspectionTool, SegmentationTool


def artifact(
    artifact_id: str,
    label: str,
    *,
    source_prompt: str | None = None,
    voxtell_prompt: str | None = None,
):
    metadata = {}
    if source_prompt is not None:
        metadata["source_prompt"] = source_prompt
    if voxtell_prompt is not None:
        metadata["voxtell_prompt"] = voxtell_prompt
    return SimpleNamespace(
        artifact_id=artifact_id,
        kind="mask",
        label=label,
        metadata=metadata,
    )


class MaskPromptAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = MaskInspectionTool(SimpleNamespace())

    def test_exact_source_prompt_resolves_normalized_mask(self) -> None:
        expected = artifact(
            "artifact_000000000001",
            "left eye",
            source_prompt="eye_L",
            voxtell_prompt="left eye",
        )
        # A newer artifact has the same canonical display name but not the exact
        # source prompt. Exact provenance must win before canonical fallback.
        canonical_only = artifact(
            "artifact_000000000002",
            "Eye-L",
            source_prompt="different source",
            voxtell_prompt="left eye",
        )
        case = SimpleNamespace(artifacts=[expected, canonical_only])

        self.assertIs(self.tool._find(case, "eye_L"), expected)

    def test_canonical_source_and_display_aliases_resolve(self) -> None:
        source_alias = artifact(
            "artifact_000000000001",
            "ocular globe",
            source_prompt="eye_L",
            voxtell_prompt="ocular globe",
        )
        display_alias = artifact(
            "artifact_000000000002",
            "Kidney_R",
            source_prompt="right kidney",
            voxtell_prompt="right kidney",
        )
        case = SimpleNamespace(artifacts=[source_alias, display_alias])

        self.assertIs(self.tool._find(case, "eye-left"), source_alias)
        self.assertIs(self.tool._find(case, "right kidney"), display_alias)

    def test_canonical_lookup_does_not_use_partial_names(self) -> None:
        left_eye = artifact(
            "artifact_000000000001",
            "left eye",
            source_prompt="eye_L",
            voxtell_prompt="left eye",
        )
        case = SimpleNamespace(artifacts=[left_eye])

        self.assertIsNone(self.tool._find(case, "eye"))


class PromptInventoryTests(unittest.TestCase):
    def test_inventory_exposes_prompt_provenance(self) -> None:
        with TemporaryDirectory() as temp:
            store = ResearchStore(Path(temp))
            case = store.create_case("image.nii.gz", BytesIO(b"image"))
            artifact_id, mask_path = store.allocate_artifact_path(
                case.case_id, ".nii.gz"
            )
            mask_path.write_bytes(b"mask")
            mask = store.register_artifact(
                case.case_id,
                artifact_id,
                mask_path,
                kind="mask",
                label="left eye",
                media_type="application/gzip",
                metadata={
                    "source_prompt": "eye_L",
                    "voxtell_prompt": "left eye",
                    "prompt_category": "anatomical_concept",
                    "prompt_exploratory": True,
                },
            )

            inventory = {
                item["artifact_id"]: item
                for item in store.artifact_inventory(case.case_id)
            }

            self.assertEqual(inventory[mask.artifact_id]["source_prompt"], "eye_L")
            self.assertEqual(inventory[mask.artifact_id]["voxtell_prompt"], "left eye")
            self.assertEqual(
                inventory[mask.artifact_id]["prompt_category"],
                "anatomical_concept",
            )
            self.assertTrue(
                inventory[mask.artifact_id]["prompt_exploratory"]
            )


class BlindPromptPersistenceTests(unittest.TestCase):
    def test_voxtell_receives_carcinoma_but_viewer_keeps_user_label(self) -> None:
        class Backend:
            name = "fake-voxtell"
            version = "test"

            def __init__(self):
                self.prompts = []

            def segment(self, image_path, prompts):
                del image_path
                self.prompts = list(prompts)
                volume = SimpleNamespace(
                    data=np.arange(8, dtype=np.float32).reshape((2, 2, 2)),
                    spacing=(1.0, 1.0, 1.0),
                    affine=np.eye(4),
                )
                mask = np.zeros((2, 2, 2), dtype=bool)
                mask[0, 0, 0] = True
                return volume, [mask]

        class Store:
            def __init__(self):
                self.artifacts = []
                self.counter = 0

            def get_case(self, case_id):
                return SimpleNamespace(
                    case_id=case_id,
                    image=SimpleNamespace(artifact_id="image"),
                )

            def artifact_path(self, artifact):
                del artifact
                return Path("/tmp/fake-image.nii.gz")

            def allocate_artifact_path(self, case_id, suffix):
                self.counter += 1
                return (
                    f"artifact_{self.counter:012d}",
                    Path(f"/tmp/fake-artifact-{self.counter}{suffix}"),
                )

            def register_artifact(
                self,
                case_id,
                artifact_id,
                path,
                kind,
                label,
                media_type,
                metadata,
            ):
                del path
                ref = ArtifactRef(
                    artifact_id=artifact_id,
                    case_id=case_id,
                    kind=kind,
                    label=label,
                    media_type=media_type,
                    sha256="0" * 64,
                    metadata=metadata,
                )
                self.artifacts.append(ref)
                return ref

        reason = "Exploratory blind report-language prompt; not a diagnosis"
        normalization = trusted_prompt_normalizations(
            ["carcinoma"],
            PromptCategory.PATHOLOGICAL_STRUCTURE,
            source_texts=["tumor"],
            exploratory=True,
            reason=reason,
        ).items
        request = SegmentRequest(
            case_id="case_000000000001",
            structures=["carcinoma"],
            purpose="segment tumor",
            target_type=TargetType.GTV,
            prompt_normalizations=normalization,
        )
        backend = Backend()
        store = Store()

        imaging = ModuleType("backend.imaging")
        imaging.mask_geometry = lambda mask: (
            int(mask.sum()),
            [[0, 0, 0], [0, 0, 0]],
            [0.0, 0.0, 0.0],
        )
        imaging.save_mask = lambda mask, volume, path: None
        imaging.render_overlays = lambda volume, mask, slices: []
        with patch.dict(sys.modules, {"backend.imaging": imaging}):
            result, _ = SegmentationTool(store, backend).run(request)

        self.assertEqual(backend.prompts, ["carcinoma"])
        self.assertEqual(result.measurements[0].structure, "tumor")
        mask = result.measurements[0].mask
        self.assertIsNotNone(mask)
        self.assertEqual(mask.label, "tumor")
        self.assertEqual(mask.metadata["source_prompt"], "tumor")
        self.assertEqual(mask.metadata["voxtell_prompt"], "carcinoma")
        self.assertEqual(
            mask.metadata["prompt_category"],
            "pathological_structure",
        )
        self.assertEqual(
            mask.metadata["prompt_normalized_by"],
            "trusted_workflow",
        )
        self.assertTrue(mask.metadata["prompt_exploratory"])
        self.assertEqual(mask.metadata["prompt_fallback_reason"], reason)


if __name__ == "__main__":
    unittest.main()
