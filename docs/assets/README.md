# Documentation artwork

`architecture.svg` is the editable, code-generated system diagram used by the
repository README. `architecture.png` is its 2× raster export (3600 × 2864).
The illustration describes `segagent-research-v2/`; it does not describe the
legacy application or invent the internal network architecture of either model.

## Regenerate

From the repository root:

```bash
python docs/assets/generate_architecture.py
```

The SVG generator requires only the Python standard library. To export the PNG
as well, optionally install CairoSVG in a documentation environment:

```bash
python -m pip install cairosvg
python docs/assets/generate_architecture.py --png
```

CairoSVG is an artwork tool, not an application dependency. Keep both exports in
version control after a diagram change and inspect the rendered PNG for clipping,
line crossings, and legibility.

## Architecture sources

The diagram follows the research implementation in `backend/workflow.py`,
`planner.py`, `tools.py`, `expert.py`, `knowledge.py`, `storage.py`, `api.py`,
`mcp_server.py`, and `a2a.py`, plus the frontend's NDJSON stream handling.
Solid arrows show actions or returned evidence. Dashed arrows show resource
access and auxiliary integrations. The mask-review gate belongs to the
segmentation path; QC treats expert masks as references rather than ground truth.
