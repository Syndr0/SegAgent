from __future__ import annotations

from .prompts import (
    dedupe_prompt_normalizations,
    fallback_prompt_normalizations,
    is_generic_pathology_prompt,
)
from .schemas import (
    PromptCategory,
    PromptNormalizationBatch,
    SegmentRequest,
    TargetType,
)
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
    """Normalize and segment prompts in an already registered local NIfTI case."""
    services = get_services()
    try:
        normalized = services.planner.normalize_prompts(structures)
        if not isinstance(normalized, PromptNormalizationBatch):
            normalized = PromptNormalizationBatch.model_validate(normalized)
    except Exception as exc:
        normalized = fallback_prompt_normalizations(
            structures,
            reason=f"{type(exc).__name__}: {exc}"[:300],
        )
    normalized = dedupe_prompt_normalizations(normalized)
    categories = {item.category for item in normalized.items}
    if any(
        is_generic_pathology_prompt(item.voxtell_prompt)
        for item in normalized.items
    ):
        raise ValueError(
            "A generic tumor/lesion prompt needs spatial context before segmentation"
        )
    if PromptCategory.ANATOMICAL_CONCEPT in categories and len(categories) > 1:
        raise ValueError(
            "Mixed anatomical and target prompts must be sent as separate requests"
        )
    target_type = (
        TargetType.ORGAN
        if categories == {PromptCategory.ANATOMICAL_CONCEPT}
        else (
            TargetType.GTV
            if categories.issubset(
                {
                    PromptCategory.PATHOLOGICAL_STRUCTURE,
                    PromptCategory.CLINICAL_DESCRIPTION,
                }
            )
            else TargetType.UNKNOWN
        )
    )
    result, _ = services.segmentation_tool.run(
        SegmentRequest(
            case_id=case_id,
            structures=[item.voxtell_prompt for item in normalized.items],
            purpose=purpose,
            target_type=target_type,
            prompt_normalizations=normalized.items,
        )
    )
    return result.model_dump(mode="json")


@mcp.tool()
def run_contour_qc(case_id: str, site: str | None = None) -> dict:
    """Check a case's uploaded contours; the model contour is a second opinion, not truth."""
    report, _ = get_services().qc_tool.run(case_id, site)
    return report.model_dump(mode="json")


@mcp.resource("segagent://cases/{case_id}")
def case_metadata(case_id: str) -> str:
    """Read de-identified case and artifact metadata without returning pixel data."""
    return get_services().store.get_case(case_id).model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
