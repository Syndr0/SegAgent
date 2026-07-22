from __future__ import annotations

from .schemas import SegmentRequest
from .service import get_services


try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - dependency message for optional entry point
    raise SystemExit("Install the research requirements to run the MCP server") from exc


mcp = FastMCP("SegAgent Research v2")


@mcp.tool()
def lookup_oar_protocol(query: str) -> dict:
    """Return a curated radiotherapy-site protocol and source-grounded passages."""
    _, observation = get_services().protocol_tool.run(query)
    return observation.model_dump(mode="json")


@mcp.tool()
def segment_volume(case_id: str, structures: list[str], purpose: str) -> dict:
    """Segment named structures in an already registered, local NIfTI case."""
    result, _ = get_services().segmentation_tool.run(
        SegmentRequest(case_id=case_id, structures=structures, purpose=purpose)
    )
    return result.model_dump(mode="json")


@mcp.tool()
def run_contour_qc(case_id: str) -> dict:
    """Audit the contours registered to a case; expert masks are references, not truth."""
    report, _ = get_services().qc_tool.run(case_id)
    return report.model_dump(mode="json")


@mcp.resource("segagent://cases/{case_id}")
def case_metadata(case_id: str) -> str:
    """Read de-identified case and artifact metadata without returning pixel data."""
    return get_services().store.get_case(case_id).model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")

