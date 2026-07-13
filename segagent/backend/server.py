import os
# Set CUDA alloc conf to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import json
import shutil
import subprocess
import tempfile
import zipfile
from typing import Annotated

import nibabel as nib
import numpy as np
import pydicom
import torch
import gc
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from rt_utils import RTStructBuilder

from voxtell.inference.predictor import VoxTellPredictor
from agent import SegAgent
from knowledge_base import OARKnowledgeBase
from dicom_sessions import (
    create_session,
    get_session_dicom_dir,
    cleanup_session,
    cleanup_expired,
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models/voxtell_v1.1")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading VoxTell model from {MODEL_DIR} on {DEVICE}...")
try:
    predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
    # Optimization: move prediction to CPU to save VRAM
    predictor.perform_everything_on_device = False
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

# SegAgent (Qwen2.5-VL planner + VoxTell expert). The LLM is loaded lazily on
# the first /chat request so the server still boots quickly.
MASK_DIR = os.path.join(tempfile.gettempdir(), "segagent_masks")

# OAR knowledge base for the lookup_oar tool. Reuse VoxTell's already-loaded
# Qwen3-Embedding-4B for semantic retrieval (no second embedding model).
KB_PATH = os.path.join(BASE_DIR, "knowledge", "oar_protocols.json")
kb = None
if predictor is not None:
    def _embed(texts):
        return predictor.embed_text_prompts(texts)[0].float().cpu().numpy()
    try:
        kb = OARKnowledgeBase(KB_PATH, embed_fn=_embed)
        print(f"Loaded OAR knowledge base ({len(kb.protocols)} protocols).")
    except Exception as e:
        print(f"Warning: could not load OAR knowledge base: {e}")

agent = None
if predictor is not None:
    agent = SegAgent(
        predictor=predictor,
        write_seg=NibabelIOWithReorient().write_seg,
        mask_dir=MASK_DIR,
        device=DEVICE,
        knowledge_base=kb,
    )


# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------

def _reorient_nifti_mask_to_dicom(nii_path: str, sorted_series_data: list) -> np.ndarray:
    """Reorient a NIfTI mask to match the DICOM pixel grid expected by rt-utils.

    rt-utils expects mask shape (Columns, Rows, num_slices) where the axes
    correspond to the DICOM pixel grid, not the NIfTI RAS voxel grid.
    dcm2niix reorients to RAS, so we must undo that by computing the axis
    permutation and flips from the NIfTI affine and the DICOM geometry.
    """
    nii = nib.load(nii_path)
    mask = np.asanyarray(nii.dataobj).astype(bool)
    affine = nii.affine

    ref_dcm = sorted_series_data[0]

    # DICOM ImageOrientationPatient in LPS coordinates
    iop = np.array(ref_dcm.ImageOrientationPatient, dtype=float)
    row_cosine_lps = iop[:3]   # direction of increasing column index
    col_cosine_lps = iop[3:]   # direction of increasing row index

    # Slice direction from sorted series positions
    if len(sorted_series_data) > 1:
        pos0 = np.array(sorted_series_data[0].ImagePositionPatient, dtype=float)
        pos1 = np.array(sorted_series_data[1].ImagePositionPatient, dtype=float)
        slice_dir_lps = pos1 - pos0
        slice_dir_lps = slice_dir_lps / np.linalg.norm(slice_dir_lps)
    else:
        slice_dir_lps = np.cross(row_cosine_lps, col_cosine_lps)

    # Convert DICOM LPS directions to RAS (flip L→R and P→A)
    lps_to_ras = np.array([-1, -1, 1], dtype=float)

    # rt-utils mask axes in RAS physical coordinates:
    #   axis 0 (Columns) → col_cosine direction (row-index direction)
    #   axis 1 (Rows)    → row_cosine direction (column-index direction)
    #   axis 2 (Slices)  → ascending slice position direction
    dicom_axes_ras = np.column_stack([
        col_cosine_lps * lps_to_ras,
        row_cosine_lps * lps_to_ras,
        slice_dir_lps * lps_to_ras,
    ])

    # NIfTI voxel axes directions in RAS (affine columns, normalized)
    nii_axes_ras = np.zeros((3, 3))
    for ax in range(3):
        v = affine[:3, ax]
        nii_axes_ras[:, ax] = v / np.linalg.norm(v)

    # Correlation: corr[nii_ax, dicom_ax] ≈ ±1 when axes correspond
    corr = nii_axes_ras.T @ dicom_axes_ras

    # For each DICOM axis find the matching NIfTI axis and flip direction
    perm = []
    flips = []
    for dicom_ax in range(3):
        abs_corr = np.abs(corr[:, dicom_ax])
        nii_ax = int(np.argmax(abs_corr))
        perm.append(nii_ax)
        flips.append(bool(corr[nii_ax, dicom_ax] < 0))

    # Apply permutation then flips
    result = np.transpose(mask, perm)
    for ax in range(3):
        if flips[ax]:
            result = np.flip(result, axis=ax)

    return np.ascontiguousarray(result)


def _find_dicom_dir(root: str) -> str | None:
    """Walk extracted zip to find a directory containing DICOM files."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip macOS resource fork directories and hidden directories
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith("._")]
        for fname in filenames:
            # Skip macOS resource fork files
            if fname.startswith("._"):
                continue
            fpath = os.path.join(dirpath, fname)
            # Check .dcm extension
            if fname.lower().endswith(".dcm"):
                return dirpath
            # Try reading extensionless files as DICOM
            try:
                pydicom.dcmread(fpath, stop_before_pixels=True)
                return dirpath
            except Exception:
                continue
    return None


def convert_dicom_to_nifti(dicom_dir: str, output_dir: str) -> str:
    """Run dcm2niix and return path to the largest output .nii.gz."""
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "converted", "-o", output_dir, dicom_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nifti_files:
        raise RuntimeError("dcm2niix produced no NIfTI output")

    # Pick the largest file (main series, not scouts)
    return max(nifti_files, key=os.path.getsize)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()]
):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, image.filename)

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        print(f"Processing {image.filename} with prompt: '{prompt}'")

        try:
            # Force GC before inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load image using nnUNet's IO to handle reorientation if needed
            # predictable behavior with nibabel
            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([input_path])

            # Run inference
            # segmentations shape: (num_prompts, X, Y, Z)
            segmentations = predictor.predict_single_image(img, [prompt])

            # We only have one prompt, so take the first result
            seg_result = segmentations[0]

            # Save output
            output_filename = f"segmentation_{image.filename}"
            output_path = os.path.join(temp_dir, output_filename)

            reader_writer.write_seg(seg_result, output_path, props)

            # Save to a persistent temp location to return it
            system_tmp = tempfile.gettempdir()
            final_output_path = os.path.join(system_tmp, f"voxtell_output_{os.urandom(8).hex()}.nii.gz")
            shutil.copy(output_path, final_output_path)

            return FileResponse(
                final_output_path,
                media_type="application/gzip",
                filename=output_filename,
                background=None
            )

        except Exception as e:
            print(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert")
async def convert_dicom(file: Annotated[UploadFile, File()]):
    """Accept a zipped DICOM folder, convert to NIfTI, return the file + session ID."""
    cleanup_expired()

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file containing DICOM data.")

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, file.filename)
        with open(zip_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Extract zip
        extract_dir = os.path.join(temp_dir, "extracted")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file.")

        # Find DICOM directory
        dicom_dir = _find_dicom_dir(extract_dir)
        if dicom_dir is None:
            raise HTTPException(status_code=400, detail="No DICOM files found in the uploaded zip.")

        # Persist DICOM series for later RTSTRUCT export
        session_id = create_session(dicom_dir)

        # Convert to NIfTI
        nifti_out_dir = os.path.join(temp_dir, "nifti")
        os.makedirs(nifti_out_dir)
        try:
            nifti_path = convert_dicom_to_nifti(dicom_dir, nifti_out_dir)
        except RuntimeError as exc:
            cleanup_session(session_id)
            raise HTTPException(status_code=500, detail=str(exc))

        # Copy NIfTI to persistent temp so FileResponse can serve it
        system_tmp = tempfile.gettempdir()
        final_path = os.path.join(system_tmp, f"voxtell_converted_{os.urandom(8).hex()}.nii.gz")
        shutil.copy(nifti_path, final_path)

        print(f"DICOM converted: session={session_id}, nifti={final_path}")

        response = FileResponse(
            final_path,
            media_type="application/gzip",
            filename="converted.nii.gz",
            background=None,
        )
        response.headers["X-Session-Id"] = session_id
        return response


@app.post("/export-rtstruct")
async def export_rtstruct(
    session_id: Annotated[str, Form()],
    structure_names: Annotated[str, Form()],  # JSON array of names
    segmentation_files: list[UploadFile] = File(...),
):
    """Build an RTSTRUCT from the stored DICOM series and uploaded segmentation masks."""
    dicom_dir = get_session_dicom_dir(session_id)
    if dicom_dir is None:
        raise HTTPException(status_code=404, detail="DICOM session not found or expired.")

    try:
        names: list[str] = json.loads(structure_names)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="structure_names must be a JSON array of strings.")

    if len(names) != len(segmentation_files):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(names)} names but {len(segmentation_files)} segmentation files.",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_dir)

        for seg_file, name in zip(segmentation_files, names):
            seg_path = os.path.join(temp_dir, seg_file.filename or "seg.nii.gz")
            with open(seg_path, "wb") as buf:
                shutil.copyfileobj(seg_file.file, buf)

            mask = _reorient_nifti_mask_to_dicom(seg_path, rtstruct.series_data)

            rtstruct.add_roi(mask=mask, name=name)

        output_path = os.path.join(temp_dir, "rtstruct.dcm")
        rtstruct.save(output_path)

        system_tmp = tempfile.gettempdir()
        final_path = os.path.join(system_tmp, f"voxtell_rtstruct_{os.urandom(8).hex()}.dcm")
        shutil.copy(output_path, final_path)

        print(f"RTSTRUCT exported: session={session_id}, structures={names}")

        return FileResponse(
            final_path,
            media_type="application/dicom",
            filename="rtstruct.dcm",
            background=None,
        )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a DICOM session and its files."""
    cleanup_session(session_id)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# SegAgent chat (reasoning-segmentation loop)
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(
    image: Annotated[UploadFile, File()],
    question: Annotated[str, Form()],
):
    """Run the reasoning-segmentation agent and stream events as NDJSON.

    Each streamed line is a JSON object of one of these types:
      thinking     -> {step, text}         a chain-of-thought step
      action       -> {step, prompt}       a call to the VoxTell expert
      observation  -> {step, prompt, text} the expert's returned statistics
      mask         -> {step, prompt, mask_id}  a produced mask (fetch via /mask)
      answer       -> {text}               the final answer
      error        -> {text}
    """
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not available (model not loaded).")

    # Persist the uploaded volume; the generator reads it lazily while streaming.
    tmp_dir = tempfile.mkdtemp(prefix="segagent_chat_")
    input_path = os.path.join(tmp_dir, image.filename or "input.nii.gz")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    def event_stream():
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([input_path])
            for event in agent.run(img, props, question):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "text": str(e)}) + "\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/mask/{mask_id}")
async def get_mask(mask_id: str):
    """Serve a mask produced by the agent so the viewer can overlay it."""
    # Guard against path traversal.
    safe = "".join(c for c in mask_id if c.isalnum())
    mask_path = os.path.join(MASK_DIR, f"{safe}.nii.gz")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask not found or expired.")
    return FileResponse(mask_path, media_type="application/gzip",
                        filename=f"segagent_{safe}.nii.gz")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
