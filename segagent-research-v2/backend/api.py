from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .a2a import router as a2a_router
from .imaging import load_mask, load_volume
from .schemas import ApprovalDecision
from .service import get_services


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


app = FastAPI(
    title="SegAgent Research v2",
    version="0.2.0-research",
    description="Stateful research API for typed 3D segmentation-agent experiments.",
)
settings = get_services().settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Run-Id"],
)
app.include_router(a2a_router)


def ndjson(items):
    for item in items:
        yield item.model_dump_json() + "\n"


@app.get("/api/health")
def health() -> dict:
    services = get_services()
    return {
        "status": "ok",
        "planner": services.settings.planner,
        "voxtellConfigured": services.settings.voxtell_model is not None,
        "researchOnly": True,
    }


@app.post("/api/cases")
def create_case(
    image: Annotated[UploadFile, File()],
    contours: Annotated[list[UploadFile] | None, File()] = None,
):
    name = image.filename or "image.nii.gz"
    if not name.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="The case image must be NIfTI (.nii/.nii.gz).")
    contour_files = contours or []
    contour_names = [contour.filename or "contour.nii.gz" for contour in contour_files]
    invalid = [
        contour_name
        for contour_name in contour_names
        if not contour_name.lower().endswith((".nii", ".nii.gz"))
    ]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Contour files must be NIfTI (.nii/.nii.gz): {invalid[0]}",
        )
    services = get_services()
    try:
        record = services.store.create_case(name, image.file)
        services.store.add_contours(
            record.case_id,
            [
                (contour_name, contour.file)
                for contour, contour_name in zip(contour_files, contour_names, strict=True)
            ],
        )
        return services.store.get_case(record.case_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    try:
        return get_services().store.get_case(case_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc


@app.post("/api/cases/{case_id}/contours")
def add_case_contours(
    case_id: str,
    contours: Annotated[list[UploadFile], File()],
):
    """Add contour masks to an existing case.

    Keeping this separate from case creation lets the UI create a case as soon
    as its image is selected and attach QC contours later.
    """
    if not contours:
        raise HTTPException(status_code=400, detail="Select at least one contour file.")
    names = [contour.filename or "contour.nii.gz" for contour in contours]
    invalid = [name for name in names if not name.lower().endswith((".nii", ".nii.gz"))]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Contour files must be NIfTI (.nii/.nii.gz): {invalid[0]}",
        )
    services = get_services()
    try:
        services.store.get_case(case_id)
        services.store.add_contours(
            case_id,
            [
                (name, contour.file)
                for contour, name in zip(contours, names, strict=True)
            ],
        )
        return services.store.get_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/cases/{case_id}/image")
def get_case_image(case_id: str):
    try:
        services = get_services()
        case = services.store.get_case(case_id)
        path = services.store.artifact_path(case.image)
        return FileResponse(path, media_type=case.image.media_type, filename=case.image.label)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Case image not found") from exc


@app.get("/api/cases/{case_id}/artifacts/{artifact_id}")
def get_artifact(case_id: str, artifact_id: str):
    try:
        ref, path = get_services().store.get_artifact(case_id, artifact_id)
        return FileResponse(path, media_type=ref.media_type, filename=ref.label)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc


@app.post("/api/cases/{case_id}/edited-mask")
def upload_edited_mask(
    case_id: str,
    structure: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    derived_from: Annotated[str | None, Form()] = None,
    target_type: Annotated[str, Form()] = "unknown",
):
    """Register a reviewer's edited contour after validating it against the case grid.

    The edited mask must sit on the same image grid as the case (NiiVue exports
    on the background volume, so this holds by construction); a mismatch is
    rejected rather than silently corrupting the evidence.
    """
    name = file.filename or "edited.nii.gz"
    if not name.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Edited mask must be NIfTI (.nii/.nii.gz).")
    services = get_services()
    store = services.store
    try:
        case = store.get_case(case_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Case not found") from exc
    reference = load_volume(store.artifact_path(case.image))
    suffix = ".nii.gz" if name.lower().endswith(".nii.gz") else ".nii"
    artifact_id, path = store.allocate_artifact_path(case_id, suffix)
    with path.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            handle.write(chunk)
    try:
        load_mask(path, reference)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"Edited mask does not match the case grid: {exc}"
        ) from exc
    label = " ".join(str(structure).strip().split()) or "edited contour"
    return store.register_artifact(
        case_id,
        artifact_id,
        path,
        "mask",
        label,
        "application/gzip" if suffix == ".nii.gz" else "application/octet-stream",
        {
            "source_model": "human_edit",
            "source_version": "1",
            "derived_from": derived_from,
            "target_type": target_type,
        },
    )


@app.post("/api/cases/{case_id}/runs")
def start_run(case_id: str, request: RunRequest):
    try:
        run_id, stream = get_services().workflow.start(case_id, request.question)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = StreamingResponse(ndjson(stream), media_type="application/x-ndjson")
    response.headers["X-Run-Id"] = run_id
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, decision: ApprovalDecision):
    try:
        stream = get_services().workflow.resume(run_id, decision)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(ndjson(stream), media_type="application/x-ndjson")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        services = get_services()
        run = services.store.get_run(run_id)
        return {
            "run": run.model_dump(mode="json"),
            "events": [item.model_dump(mode="json") for item in services.store.events(run_id)],
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc
