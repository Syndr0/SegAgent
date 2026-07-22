from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import get_services


router = APIRouter(tags=["A2A research adapter"])


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


def rpc_result(request_id: str | int | None, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def rpc_error(request_id: str | int | None, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


@router.get("/.well-known/agent-card.json")
def agent_card() -> dict:
    settings = get_services().settings
    return {
        "name": "SegAgent Research v2",
        "description": "Evidence-first 3D medical segmentation and contour-QC research agent.",
        "version": "0.2.0-research",
        "url": f"{settings.api_base_url}/a2a",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json", "application/gzip"],
        "skills": [
            {
                "id": "segment-3d",
                "name": "3D text-prompted segmentation",
                "description": "Segments named structures in an existing case.",
                "tags": ["medical-imaging", "segmentation", "NIfTI", "VoxTell"],
                "examples": ["Segment the liver and both kidneys in case case_..."],
            },
            {
                "id": "contour-qc",
                "name": "Contour quality control",
                "description": "Audits registered contours using geometry and expert disagreement.",
                "tags": ["radiotherapy", "quality-control", "human-review"],
                "examples": ["Audit the uploaded contours for case case_..."],
            },
        ],
        "metadata": {
            "researchOnly": True,
            "caseIdRequired": True,
            "conformanceStatus": "research-adapter-not-certified",
        },
    }


@router.post("/a2a")
def a2a_rpc(request: JsonRpcRequest) -> dict:
    services = get_services()
    try:
        if request.method == "message/send":
            message = request.params.get("message", {})
            metadata = message.get("metadata", {})
            case_id = metadata.get("caseId") or request.params.get("caseId")
            if not isinstance(case_id, str):
                return rpc_error(request.id, -32602, "message metadata.caseId is required")
            parts = message.get("parts", [])
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
            question = "\n".join(item for item in text_parts if item).strip()
            if not question:
                return rpc_error(request.id, -32602, "a text message part is required")
            run_id, stream = services.workflow.start(case_id, question)
            events = [item.model_dump(mode="json") for item in stream]
            run = services.store.get_run(run_id)
            state = "completed" if run.status == "completed" else "input-required"
            return rpc_result(
                request.id,
                {
                    "id": run_id,
                    "contextId": case_id,
                    "status": {"state": state},
                    "artifacts": [
                        {
                            "artifactId": f"artifact_{uuid.uuid4().hex[:16]}",
                            "name": "SegAgent event trace",
                            "parts": [{"data": {"events": events}, "mediaType": "application/json"}],
                        }
                    ],
                },
            )
        if request.method == "tasks/get":
            task_id = request.params.get("id")
            run = services.store.get_run(str(task_id))
            return rpc_result(
                request.id,
                {
                    "id": run.run_id,
                    "contextId": run.case_id,
                    "status": {"state": run.status},
                    "history": [
                        item.model_dump(mode="json") for item in services.store.events(run.run_id)
                    ],
                },
            )
        return rpc_error(request.id, -32601, f"unsupported method: {request.method}")
    except (ValueError, FileNotFoundError) as exc:
        return rpc_error(request.id, -32602, str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="A2A research adapter failed") from exc

