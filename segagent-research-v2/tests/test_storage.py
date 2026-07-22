from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.storage import ResearchStore


class StorageTests(unittest.TestCase):
    def test_cases_are_isolated(self) -> None:
        with TemporaryDirectory() as temp:
            store = ResearchStore(Path(temp))
            first = store.create_case("first.nii.gz", BytesIO(b"first-image"))
            second = store.create_case("second.nii.gz", BytesIO(b"second-image"))

            self.assertNotEqual(first.case_id, second.case_id)
            self.assertEqual(store.artifact_path(first.image).read_bytes(), b"first-image")
            self.assertEqual(store.artifact_path(second.image).read_bytes(), b"second-image")

    def test_invalid_case_id_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            store = ResearchStore(Path(temp))
            with self.assertRaises(ValueError):
                store.case_dir("../../outside")

    def test_artifact_provenance_is_persisted(self) -> None:
        with TemporaryDirectory() as temp:
            store = ResearchStore(Path(temp))
            case = store.create_case("image.nii.gz", BytesIO(b"image"))
            artifact_id, source = store.allocate_artifact_path(case.case_id, ".nii.gz")
            source.write_bytes(b"mask")

            artifact = store.register_artifact(
                case.case_id,
                artifact_id,
                source,
                kind="mask",
                label="synthetic mask",
                media_type="application/gzip",
                metadata={"source_model": "fake", "source_version": "test"},
            )

            persisted, persisted_path = store.get_artifact(case.case_id, artifact.artifact_id)
            self.assertEqual(persisted.sha256, artifact.sha256)
            self.assertEqual(persisted.metadata["source_model"], "fake")
            self.assertEqual(persisted_path.read_bytes(), b"mask")

    def test_uncompressed_contour_keeps_nii_extension(self) -> None:
        with TemporaryDirectory() as temp:
            store = ResearchStore(Path(temp))
            case = store.create_case("image.nii", BytesIO(b"image"))
            contour = store.add_contour(
                case.case_id, "left_kidney.nii", BytesIO(b"contour")
            )

            self.assertEqual(store.artifact_path(contour).suffix, ".nii")
            self.assertEqual(contour.media_type, "application/octet-stream")


if __name__ == "__main__":
    unittest.main()
