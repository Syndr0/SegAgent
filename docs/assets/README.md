# Documentation artwork

`architecture.svg` is the editable, code-generated system diagram used by the
repository README. `architecture.png` is its 2× raster export (3600 × 3328).
The illustration describes **SegAgent**, the application in `segagent/`.
The diagram shows the system and model adapters, without inventing the internal
network architecture of either model.

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

The diagram follows the implementation in `segagent/backend/workflow.py`,
`segagent/backend/intent.py`, `segagent/backend/dialogue.py`,
`segagent/backend/prompts.py`, `segagent/backend/planner.py`,
`segagent/backend/tools.py`, `segagent/backend/verifier.py`,
`segagent/backend/qc.py`, `segagent/backend/expert.py`,
`segagent/backend/batching.py`, `segagent/backend/knowledge.py`,
`segagent/backend/storage.py`, `segagent/backend/api.py`,
`segagent/backend/mcp_server.py`, and `segagent/backend/a2a.py`, plus the
frontend's NDJSON stream handling in `segagent/frontend/`.
Solid arrows show actions or returned evidence. Dashed arrows show resource
access and auxiliary integrations. Each case has one continuing conversation;
clarification replies resume the suspended turn. Segmentation has no universal
mask-approval gate. QC uses ROI roles and typed evidence, while expert masks
remain comparison references rather than ground truth. These software checks do
not constitute clinical validation.
