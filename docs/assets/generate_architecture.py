#!/usr/bin/env python3
"""Regenerate the SegAgent architecture illustration.

The SVG uses only Python's standard library and remains fully editable.
    python docs/assets/generate_architecture.py

Optionally render a 2x PNG with the documentation-only CairoSVG package:
    python -m pip install cairosvg
    python docs/assets/generate_architecture.py --png

Neither command changes the application's runtime dependencies.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
W, H = 1800, 1664
INK = "#172B46"
MUTED = "#52657C"
TEAL = "#087F8C"
PURPLE = "#6855B3"
LINE = "#CDD8E5"
PARTS: list[str] = []


def add(value: str) -> None:
    PARTS.append(value)


def rect(x, y, w, h, *, fill="white", stroke=LINE, r=16, extra=""):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" {extra}/>')


def text(x, y, value, size=22, color=INK, weight=400, *, anchor="start", extra=""):
    add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" {extra}>{escape(value)}</text>')


def line(x1, y1, x2, y2, color=LINE, width=1.5):
    add(f'<path d="M{x1} {y1}H{x2}" stroke="{color}" stroke-width="{width}"/>'
        if y1 == y2 else
        f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" stroke-width="{width}"/>')


def arrow(path, color=TEAL, *, dash=False, marker="teal", width=2.4):
    dash_attribute = 'stroke-dasharray="7 6"' if dash else ""
    add(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'{dash_attribute} '
        f'marker-end="url(#{marker})"/>')


def icon(kind, x, y, color=TEAL, scale=1):
    paths = {
        "layers": '<path d="m0 8 16-8 16 8-16 8Z"/><path d="m0 17 16 8 16-8M0 26l16 8 16-8"/>',
        "volume": '<path d="m1 9 15-8 15 8v18l-15 8-15-8Z"/><path d="m1 9 15 8 15-8M16 17v18"/>',
        "screen": '<rect x="0" y="1" width="34" height="25" rx="4"/><path d="M11 34h12M17 26v8"/>',
        "api": '<path d="m10 5-9 12 9 12m14-24 9 12-9 12M20 0l-6 34"/>',
        "plug": '<path d="M9 0v10m16-10v10M5 10h24v6a12 12 0 0 1-24 0Zm12 18v9"/>',
        "database": '<ellipse cx="17" cy="6" rx="16" ry="6"/><path d="M1 6v23c0 8 32 8 32 0V6M1 17c0 8 32 8 32 0"/>',
        "check": '<path d="M4 17l8 8L30 6"/><circle cx="17" cy="17" r="17"/>',
        "docs": '<path d="M5 0h20l8 8v28H5Zm20 0v9h8M12 16h14M12 23h14M12 30h10"/>',
    }
    add(f'<g transform="translate({x} {y}) scale({scale})" fill="none" '
        f'stroke="{color}" stroke-width="2.1" stroke-linejoin="round" '
        f'stroke-linecap="round">{paths[kind]}</g>')


def pill(x, y, w, value, *, color=TEAL, fill="#E9F5F5", size=18):
    rect(x, y, w, 30, fill=fill, stroke="none", r=15)
    text(x + w / 2, y + 21, value, size, color, 600, anchor="middle")


def render() -> str:
    PARTS.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">')
    add('<title id="title">SegAgent architecture</title>')
    add('<desc id="description">A React workspace sends messages and referenced mask '
        'IDs to FastAPI. One conversation per case provides history and checkpointed '
        'LangGraph state. Typed intent and a dialogue state machine constrain the '
        'Qwen3.5-9B planner. Actions include protocol lookup, prompt-normalized VoxTell '
        'segmentation in bounded batches, existing-mask inspection, role-aware contour '
        'quality control, clarification that resumes the same turn, and visual question '
        'answering. Tool evidence returns to planning; a metric-aware grounding verifier '
        'and evidence critic guard final answers. Seed protocols, optional guideline '
        'text, independent VoxTell inference, file-backed artifacts, JSONL messages '
        'and events, and SQLite graph checkpoints support the workflow. Live NDJSON '
        'includes thinking_delta and workflow events; a complete answer event '
        'delivers the final response. MCP exposes tools; A2A is an '
        'auxiliary research adapter. These research checks are not clinical validation.</desc>')
    add('<defs>')
    for name, color in (("teal", TEAL), ("muted", MUTED), ("purple", PURPLE)):
        add(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
            f'<path d="M0 0L8 4.5L0 9Z" fill="{color}"/></marker>')
    add('</defs>')
    add('<g font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">')
    rect(0, 0, W, H, fill="#F7F9FC", stroke="none", r=0)
    rect(30, 30, 1740, H - 60, fill="white", stroke="#E2E8F0", r=24)

    # Heading and a compact, code-native slice mark.
    icon("layers", 64, 63, scale=1.25)
    text(122, 93, "SegAgent", 42, INK, 700)
    pill(338, 64, 258, "RESEARCH SOFTWARE", color=PURPLE, fill="#F0EDFA", size=19)
    text(64, 130, "Evidence-first orchestration for 3D medical images", 25, MUTED)
    text(1736, 91, "SYSTEM ARCHITECTURE", 19, MUTED, 600, anchor="end",
         extra='letter-spacing="1.6"')

    # Entry points. Two distinct horizontal arrows encode requests and streamed results.
    rect(64, 170, 300, 150)
    icon("volume", 90, 194)
    text(138, 221, "Case inputs", 25, weight=650)
    text(90, 263, "3D NIfTI image", 22, MUTED)
    text(90, 294, "+ optional contours", 22, MUTED)
    arrow("M364 245H420")

    rect(420, 170, 460, 150)
    icon("screen", 446, 194)
    text(498, 221, "React research workspace", 25, weight=650)
    text(446, 263, "Messages + referenced mask IDs", 22, MUTED)
    text(446, 294, "Live NDJSON · viewer · edited masks", 21, MUTED)
    arrow("M880 222H936")
    arrow("M936 282H880", MUTED, marker="muted")

    rect(936, 170, 292, 150)
    icon("api", 962, 194)
    text(1014, 221, "FastAPI", 25, weight=650)
    text(962, 263, "Case + artifact API", 22, MUTED)
    text(962, 294, "Conversation streams", 22, MUTED)

    rect(1284, 170, 452, 150, fill="#FBFAFE", stroke="#DED7EF")
    icon("plug", 1310, 194, PURPLE)
    text(1362, 221, "Auxiliary agent access", 25, weight=650)
    text(1310, 263, "MCP: stdio tools + case metadata", 21, MUTED)
    text(1310, 294, "A2A: research JSON-RPC adapter", 21, MUTED)
    arrow("M1284 282H1228", PURPLE, dash=True, marker="purple")

    # The REST service enters/resumes the graph; persisted events flow back up.
    arrow("M1056 320V364")
    arrow("M1112 364V320", MUTED, marker="muted")
    text(1037, 348, "turn", 17, TEAL, anchor="end")
    text(1130, 348, "events", 17, MUTED)

    # One case owns one continuing graph thread; individual turns reset their
    # tool evidence and guards while keeping conversation history.
    rect(64, 364, 1164, 1016, fill="#F9FBFE", stroke="#C5D3E3", r=20)
    text(96, 403, "LangGraph conversation workflow", 26, weight=650)
    text(558, 403, "checkpointed · bounded steps · clarification resume", 21, MUTED)
    arrow("M1696 320V347H1210V544", PURPLE, dash=True, marker="purple")
    text(1452, 343, "MCP → tools", 17, PURPLE)
    rect(96, 436, 1100, 90, fill="#EDF3FA", stroke="#D1DFEE", r=14)
    text(124, 470, "Case conversation", 24, weight=650)
    text(124, 503, "History + viewer mask IDs", 21, MUTED)
    line(516, 454, 516, 508, "#CFDBE9")
    text(546, 470, "Typed intent", 24, weight=650)
    text(546, 503, "Task + target classification", 20, MUTED)
    line(872, 454, 872, 508, "#CFDBE9")
    text(902, 470, "Dialogue FSM", 24, weight=650)
    text(902, 503, "Stages + duplicate guards", 20, MUTED)
    arrow("M260 526V604")
    text(280, 568, "turn context", 19, MUTED)

    rect(484, 544, 736, 638, fill="#F0F6F8", stroke="#D5E4E9", r=17)
    text(508, 582, "TYPED NEXT ACTIONS", 18, TEAL, 700, extra='letter-spacing="1.5"')
    arrow("M424 742H484")
    arrow("M734 1182V1206H444V878H424", MUTED, marker="muted")
    text(600, 1233, "Observations + reply context", 20, MUTED, anchor="middle")
    arrow("M260 908V1304H508")
    text(284, 1289, "final", 20, TEAL)
    arrow("M1027 1150V1230H760V1252")
    text(1044, 1225, "visual answer", 18, TEAL)

    rect(96, 604, 328, 304, fill="white", stroke="#B9CDE4", r=16)
    text(124, 647, "Qwen planner", 28, weight=650)
    text(124, 683, "Qwen3.5-9B · default", 22, MUTED)
    line(124, 705, 396, 705, "#E0E8F0")
    text(124, 739, "Grounding views + masks", 21, MUTED)
    text(124, 771, "Planning + visual answers", 21, MUTED)
    text(124, 803, "Prompt normalization", 21, MUTED)
    text(124, 842, "Validated JSON decisions", 21, TEAL, 600)
    pill(124, 862, 273, "RulePlanner baseline available", color=MUTED,
         fill="#EFF3F8", size=16.5)

    rect(508, 604, 688, 98)
    text(536, 642, "lookup_protocol", 25, weight=650)
    text(536, 680, "Site-specific OAR list + retrieved passages", 22, MUTED)

    rect(508, 738, 688, 120)
    text(536, 777, "segment", 25, weight=650)
    text(536, 811, "Normalize prompts → VoxTell", 21, MUTED)
    text(536, 841, "Masks · measurements · overlays", 21, MUTED)
    rect(924, 766, 246, 66, fill="#F4F0FC", stroke="#D9CDEE", r=12)
    text(1047, 793, "Bounded prompt batches", 19, PURPLE, 650, anchor="middle")
    text(1047, 819, "default: 4 / chunk", 18, PURPLE, anchor="middle")

    rect(508, 894, 330, 110)
    text(536, 932, "inspect_mask", 25, weight=650)
    text(536, 965, "Existing mask measurements", 20, MUTED)
    text(536, 990, "Artifact IDs + image grid", 19, MUTED)
    rect(858, 894, 338, 110)
    text(886, 932, "run_qc", 25, weight=650)
    text(886, 965, "ROI roles + model comparison", 19, MUTED)
    text(886, 990, "Metrics + evidence records", 19, MUTED)

    rect(508, 1040, 330, 110, fill="#FAF8FE", stroke="#D9D1EB")
    text(536, 1078, "ask_user", 25, weight=650)
    text(536, 1110, "Clarification → reply → resume", 19, MUTED)
    text(536, 1135, "OAR / target / report dialogue", 18, MUTED)
    rect(858, 1040, 338, 110, fill="#FAF8FE", stroke="#D9D1EB")
    text(886, 1078, "answer_question", 25, weight=650)
    text(886, 1110, "Qwen image / mask VQA", 20, MUTED)
    text(886, 1135, "Uses recorded measurements", 19, MUTED)

    rect(508, 1252, 368, 104, fill="white", stroke="#B9CDE4", r=15)
    text(536, 1288, "Evidence critic + verifier", 24, weight=650)
    text(536, 1320, "Metric / unit / citation grounding", 19, MUTED)
    text(536, 1345, "Guarded answer or evidence fallback", 18, MUTED)
    arrow("M876 1304H936")
    rect(936, 1252, 260, 104, fill="#EAF5F5", stroke="#B8DCDD", r=15)
    text(962, 1288, "Answer + evidence", 23, TEAL, 650)
    text(962, 1320, "Recorded run events", 20, MUTED)
    text(962, 1345, "+ artifact references", 19, MUTED)

    # Shared resources: Qwen planning and VoxTell inference remain independent.
    text(1284, 582, "SHARED RESOURCES", 19, MUTED, 600, extra='letter-spacing="1.5"')
    rect(1284, 604, 452, 98, fill="#F7FAFD", stroke="#CDD9E7")
    icon("docs", 1310, 622, color=TEAL, scale=.85)
    text(1357, 642, "Local knowledge", 25, weight=650)
    text(1310, 673, "Seed protocols + optional guideline text", 20, MUTED)
    text(1310, 695, "Aliases · BM25 · semantic retrieval", 18, MUTED)
    arrow("M1196 653H1284", MUTED, dash=True, marker="muted")

    rect(1284, 738, 452, 266, fill="#F7F5FC", stroke="#D3CBE8")
    icon("volume", 1310, 756, PURPLE)
    text(1358, 783, "VoxTell expert", 28, weight=650)
    text(1310, 820, "Text-prompted 3D segmentation", 23, PURPLE, 500)
    line(1310, 839, 1710, 839, "#DED6ED")
    text(1310, 873, "RAS-aligned image + typed prompts", 20, MUTED)
    text(1310, 906, "Bounded batches → masks in case grid", 20, MUTED)
    text(1310, 944, "Shared text encoder for retrieval", 21, MUTED)
    text(1310, 981, "Independent of Qwen3.5", 20, PURPLE, 500)
    arrow("M1196 792H1284", MUTED, dash=True, marker="muted")
    arrow("M1196 952H1284", MUTED, dash=True, marker="muted")
    arrow("M1508 738V702", MUTED, dash=True, marker="muted")
    text(1526, 729, "text embeddings", 17, MUTED)

    rect(1284, 1040, 452, 110, fill="#FBFCFE", stroke="#D9E1EB", r=15)
    text(1310, 1078, "Optional observability", 25, weight=650)
    text(1310, 1110, "OpenTelemetry agent + tool spans", 21, MUTED)
    text(1310, 1135, "Identifier and count spans", 18, MUTED)
    arrow("M1228 1095H1284", MUTED, dash=True, marker="muted")
    rect(1284, 1252, 452, 104, fill="#F5F9FC", stroke="#D2DFEA", r=15)
    text(1310, 1288, "Incremental NDJSON", 25, weight=650)
    text(1310, 1320, "thinking_delta · workflow events", 21, MUTED)
    text(1310, 1345, "Final response: complete answer event", 19, MUTED)

    # Durable conversation state, traceable artifacts and explicit research scope.
    arrow("M688 1380V1436", MUTED, marker="muted")
    text(707, 1416, "persist + replay", 19, MUTED)
    rect(64, 1436, 1164, 138, fill="#F3F7FB", stroke="#C8D5E4")
    icon("database", 96, 1458, MUTED)
    text(146, 1485, "Case + conversation persistence", 27, weight=650)
    text(96, 1523, "Cases · conversation messages · masks · overlays · JSONL run events", 23, MUTED)
    text(96, 1555, "SQLite graph checkpoints · SHA-256 artifacts · typed metric evidence", 22, MUTED)
    rect(1284, 1436, 452, 138, fill="#FAF8FD", stroke="#DED7EC")
    text(1310, 1478, "RESEARCH ONLY", 18, PURPLE, 700, extra='letter-spacing="1.2"')
    text(1310, 1517, "These checks are not clinical validation.", 20, MUTED)
    text(1310, 1548, "Expert masks are references, not truth.", 20, MUTED)

    # Legend, including what dashed dependencies mean.
    arrow("M64 1612H105")
    text(120, 1619, "Workflow / user action", 18, MUTED)
    arrow("M418 1612H459", MUTED, marker="muted")
    text(474, 1619, "Evidence / events", 18, MUTED)
    arrow("M724 1612H765", MUTED, dash=True, marker="muted")
    text(780, 1619, "Resource access", 18, MUTED)
    arrow("M1027 1612H1068", PURPLE, dash=True, marker="purple")
    text(1083, 1619, "Auxiliary integration", 18, MUTED)
    text(1736, 1619, "SegAgent", 19, MUTED, anchor="end")
    add('</g></svg>')
    return "\n".join(PARTS) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also export a 2x PNG with CairoSVG")
    args = parser.parse_args()
    destination = OUT / "architecture.svg"
    destination.write_text(render(), encoding="utf-8")
    print(destination)
    if args.png:
        try:
            import cairosvg
        except ImportError as exc:
            raise SystemExit("PNG export requires the documentation-only package: pip install cairosvg") from exc
        png_path = OUT / "architecture.png"
        cairosvg.svg2png(url=str(destination), write_to=str(png_path), scale=2)
        print(png_path)


if __name__ == "__main__":
    main()
