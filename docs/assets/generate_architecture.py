#!/usr/bin/env python3
"""Regenerate the Research v2 architecture illustration.

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
W, H = 1800, 1432
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
    add('<title id="title">SegAgent Research v2 architecture</title>')
    add('<desc id="description">A React interface sends case questions and review decisions '
        'to FastAPI. A checkpointed LangGraph workflow uses the Qwen2.5-VL planner to '
        'select protocol lookup, text-prompted VoxTell segmentation with human mask review, '
        'or contour quality control. Protocol lookup uses seed protocols and optional '
        'guideline text. Typed tool observations return to the planner. '
        'An evidence critic checks the final answer. Local knowledge, the independent '
        'VoxTell expert, file-backed artifacts, JSONL events and SQLite checkpoints '
        'support the workflow. MCP exposes tools directly; the A2A research adapter '
        'uses the workflow. Results stream to the interface as NDJSON.</desc>')
    add('<defs>')
    for name, color in (("teal", TEAL), ("muted", MUTED), ("purple", PURPLE)):
        add(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
            f'orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
            f'<path d="M0 0L8 4.5L0 9Z" fill="{color}"/></marker>')
    add('</defs>')
    add('<g font-family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">')
    rect(0, 0, W, H, fill="#F7F9FC", stroke="none", r=0)
    rect(30, 30, 1740, 1372, fill="white", stroke="#E2E8F0", r=24)

    # Heading and a compact, code-native slice mark.
    icon("layers", 64, 63, scale=1.25)
    text(122, 93, "SegAgent", 42, INK, 700)
    pill(338, 64, 186, "RESEARCH V2", color=PURPLE, fill="#F0EDFA", size=19)
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
    text(446, 263, "Questions + review decisions", 22, MUTED)
    text(446, 294, "NDJSON trace · masks · overlays", 22, MUTED)
    arrow("M880 222H936")
    arrow("M936 282H880", MUTED, marker="muted")

    rect(936, 170, 292, 150)
    icon("api", 962, 194)
    text(1014, 221, "FastAPI", 25, weight=650)
    text(962, 263, "Case + artifact API", 22, MUTED)
    text(962, 294, "Run / resume streams", 22, MUTED)

    rect(1284, 170, 452, 150, fill="#FBFAFE", stroke="#DED7EF")
    icon("plug", 1310, 194, PURPLE)
    text(1362, 221, "Auxiliary agent access", 25, weight=650)
    text(1310, 263, "MCP: stdio tools + case metadata", 21, MUTED)
    text(1310, 294, "A2A: research JSON-RPC adapter", 21, MUTED)
    arrow("M1284 282H1228", PURPLE, dash=True, marker="purple")

    # The REST service enters/resumes the graph; persisted events flow back up.
    arrow("M1056 320V364")
    arrow("M1112 364V320", MUTED, marker="muted")
    text(1037, 348, "run", 17, TEAL, anchor="end")
    text(1130, 348, "events", 17, MUTED)

    # Main workflow container, its planner and the typed tool boundary.
    rect(64, 364, 1164, 784, fill="#F9FBFE", stroke="#C5D3E3", r=20)
    text(96, 403, "LangGraph workflow", 26, weight=650)
    text(376, 403, "checkpointed · bounded steps · pause / resume", 21, MUTED)
    arrow("M1696 320V347H1180V420", PURPLE, dash=True, marker="purple")
    text(1452, 343, "MCP → tools", 17, PURPLE)
    rect(484, 420, 736, 526, fill="#F0F6F8", stroke="#D5E4E9", r=17)
    text(508, 456, "TYPED TOOL EXECUTION", 18, TEAL, 700, extra='letter-spacing="1.5"')

    # Draw loop paths before the nodes so the edge ports stay visually clean.
    arrow("M424 518H462V848H508")
    arrow("M462 518H508")
    arrow("M462 668H508")
    text(447, 604, "select one action", 18, TEAL, anchor="middle",
         extra='transform="rotate(-90 447 604)"')
    arrow("M850 946V968H444V722H424", MUTED, marker="muted")
    text(770, 996, "Typed observations + artifact evidence", 20, MUTED, anchor="middle")
    arrow("M260 754V1064H508")
    text(284, 1049, "final / ask_user", 20, TEAL)

    rect(96, 450, 328, 304, fill="white", stroke="#B9CDE4", r=16)
    text(124, 493, "Qwen planner", 28, weight=650)
    text(124, 529, "Qwen2.5-VL-7B-Instruct", 21, MUTED)
    line(124, 551, 396, 551, "#E0E8F0")
    text(124, 584, "Image grounding views", 21, MUTED)
    text(124, 614, "+ prior mask overlays", 21, MUTED)
    text(124, 644, "+ typed tool observations", 21, MUTED)
    text(124, 683, "JSON → PlannerDecision", 21, TEAL, 600)
    pill(124, 708, 273, "RulePlanner baseline available", color=MUTED,
         fill="#EFF3F8", size=16.5)

    rect(508, 480, 688, 112)
    text(536, 519, "lookup_protocol", 25, weight=650)
    text(536, 557, "Site-specific OAR list + retrieved passages", 22, MUTED)

    rect(508, 630, 688, 142)
    text(536, 669, "segment", 25, weight=650)
    text(536, 709, "VoxTell masks + measurements", 21, MUTED)
    text(536, 740, "NIfTI + PNG overlay artifacts", 21, MUTED)
    arrow("M864 714H902")
    rect(902, 677, 268, 72, fill="#F4F0FC", stroke="#D9CDEE", r=12)
    text(1036, 705, "Human mask review", 21, PURPLE, 650, anchor="middle")
    text(1036, 732, "interrupt / resume", 19, PURPLE, anchor="middle")

    rect(508, 810, 688, 112)
    text(536, 849, "run_qc", 25, weight=650)
    text(536, 887, "Contour geometry + VoxTell reference disagreement", 21, MUTED)

    rect(508, 1012, 368, 104, fill="white", stroke="#B9CDE4", r=15)
    text(536, 1048, "Evidence critic", 25, weight=650)
    text(536, 1080, "Numeric + evidence checks", 21, MUTED)
    text(536, 1105, "Guarded answer or limitation", 19, MUTED)
    arrow("M876 1064H936")
    rect(936, 1012, 260, 104, fill="#EAF5F5", stroke="#B8DCDD", r=15)
    text(962, 1048, "Answer / clarify", 24, TEAL, 650)
    text(962, 1080, "Recorded run events", 20, MUTED)
    text(962, 1105, "+ artifact references", 19, MUTED)

    # Shared resources: planner and expert are deliberately independent models.
    text(1284, 403, "SHARED RESOURCES", 19, MUTED, 600, extra='letter-spacing="1.5"')
    rect(1284, 480, 452, 112, fill="#F7FAFD", stroke="#CDD9E7")
    icon("docs", 1310, 502, color=TEAL, scale=.85)
    text(1357, 519, "Local knowledge", 25, weight=650)
    text(1310, 553, "Seed protocols + optional guideline text", 20, MUTED)
    text(1310, 579, "Aliases · BM25 · semantic retrieval", 19, MUTED)
    arrow("M1196 536H1284", MUTED, dash=True, marker="muted")

    rect(1284, 630, 452, 292, fill="#F7F5FC", stroke="#D3CBE8")
    icon("volume", 1310, 653, PURPLE)
    text(1358, 680, "VoxTell expert", 28, weight=650)
    text(1310, 719, "Text-prompted 3D segmentation", 23, PURPLE, 500)
    line(1310, 740, 1710, 740, "#DED6ED")
    text(1310, 775, "Image volume + structure prompts", 21, MUTED)
    text(1310, 808, "→ binary masks in the case grid", 21, MUTED)
    text(1310, 851, "Shared text encoder for retrieval", 21, MUTED)
    text(1310, 891, "Independent of the Qwen planner", 20, PURPLE, 500)
    arrow("M1196 649H1284", MUTED, dash=True, marker="muted")
    arrow("M1196 866H1284", MUTED, dash=True, marker="muted")
    arrow("M1508 630V592", MUTED, dash=True, marker="muted")
    text(1526, 619, "text embeddings", 17, MUTED)

    rect(1284, 1012, 452, 104, fill="#FBFCFE", stroke="#D9E1EB", r=15)
    text(1310, 1049, "Optional observability", 25, weight=650)
    text(1310, 1081, "OpenTelemetry agent + tool spans", 21, MUTED)
    text(1310, 1106, "Identifier and count spans", 18, MUTED)
    arrow("M1228 1094H1284", MUTED, dash=True, marker="muted")

    # Durable evidence and explicit research semantics.
    arrow("M688 1148V1204", MUTED, marker="muted")
    text(707, 1184, "persist + replay", 19, MUTED)
    rect(64, 1204, 1164, 138, fill="#F3F7FB", stroke="#C8D5E4")
    icon("database", 96, 1226, MUTED)
    text(146, 1253, "Case-scoped persistence", 27, weight=650)
    text(96, 1291, "NIfTI inputs · masks · overlays · JSON records · append-only JSONL events", 23, MUTED)
    text(96, 1323, "LangGraph checkpoints: SQLite     |     SHA-256 artifact provenance", 22, MUTED)
    rect(1284, 1204, 452, 138, fill="#FAF8FD", stroke="#DED7EC")
    text(1310, 1246, "RESEARCH CONTRACT", 18, PURPLE, 700, extra='letter-spacing="1.2"')
    text(1310, 1285, "Human review guides continuation.", 21, MUTED)
    text(1310, 1316, "Expert masks are references, not truth.", 20, MUTED)

    # Legend, including what dashed dependencies mean.
    arrow("M64 1380H105")
    text(120, 1387, "Workflow / user action", 18, MUTED)
    arrow("M418 1380H459", MUTED, marker="muted")
    text(474, 1387, "Evidence / events", 18, MUTED)
    arrow("M724 1380H765", MUTED, dash=True, marker="muted")
    text(780, 1387, "Resource access", 18, MUTED)
    arrow("M1027 1380H1068", PURPLE, dash=True, marker="purple")
    text(1083, 1387, "Auxiliary integration", 18, MUTED)
    text(1736, 1387, "segagent-research-v2/", 19, MUTED, anchor="end")
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
