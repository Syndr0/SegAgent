# Architecture

This document describes `segagent-research-v2/`. The older `segagent/` tree is an
overlay for a separate VoxTell web-plugin checkout and has a different API and
execution model.

![SegAgent architecture](assets/architecture.svg)

## Components

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Research console | React, TypeScript, NiiVue | Upload a case and contours, display masks and run events, and submit review decisions. |
| HTTP API | `backend/api.py` | Case uploads, artifact downloads, run streams, and approval resume. |
| Planner | `backend/planner.py` | Select one typed next action using local Qwen2.5-VL or a deterministic rule baseline. |
| Workflow | `backend/workflow.py` | Execute tools, maintain run state, checkpoint, pause for mask review, and finish a run. |
| Segmentation adapter | `backend/expert.py` | Lazily load VoxTell for full-volume segmentation and reuse its text-embedding interface for retrieval. |
| Domain tools | `backend/tools.py` | Protocol lookup, segmentation, contour QC, and a final-answer evidence heuristic. |
| Case and artifact store | `backend/storage.py` | Persist case metadata, uploaded and generated files, provenance hashes, and run event logs. |
| Interoperability adapters | `backend/mcp_server.py`, `backend/a2a.py` | Expose domain tools through MCP and a limited A2A research adapter. |

## Models and evidence

The default planner is **`Qwen/Qwen2.5-VL-7B-Instruct`**, loaded through
`Qwen2_5_VLForConditionalGeneration`. It receives representative axial views,
coronal and sagittal mid-plane views, the current question, and tool observations.
It can also receive recent case overlays. It does not ingest the full 3D volume.

**VoxTell** segments the full canonicalized volume from anatomical text prompts.
The adapter also calls VoxTell's `embed_text_prompts` method for semantic
retrieval; SegAgent does not load a separate embedding model itself. VoxTell's
code and model assets must be installed separately. This repository contains no
training or fine-tuning pipeline.

A segmentation observation contains per-structure measurements, mask references,
and overlay references. Measurements include voxel count, volume in mL, mean
intensity, voxel bounding box, and voxel centroid. Empty model predictions are
reported as empty results rather than saved as nonempty masks.

## One run through the graph

1. A NIfTI upload creates a case. Optional contour masks can be attached at
   creation time or later.
2. A question creates a new run. The planner selects a validated
   `PlannerDecision`: `lookup_protocol`, `segment`, `run_qc`, `ask_user`, or
   `final`. Qwen output is schema-guided JSON with one correction attempt after
   a decoding or validation failure; it is not constrained decoding.
3. Protocol lookup and QC return observations to the planner. Segmentation
   produces measurements and artifacts, then enters the review node.
4. When segmentation generated masks, review pauses the graph by default using
   a LangGraph interrupt. An `approve`, `feedback`, or `reject` decision resumes
   the checkpoint. Feedback and rejection become additional observations;
   they do not delete or invalidate stored mask files automatically.
5. `final` or `ask_user` finishes the run after the evidence critic examines the
   proposed answer. Reaching the configured step budget also produces a final
   evidence summary. `ask_user` is a completed clarification response, not an
   additional resumable interrupt.

The rule planner removes the Qwen dependency from planning. It does **not**
replace VoxTell for segmentation or QC, and a semantic retrieval path can still
load VoxTell.

## Persistence and streaming

Graph state contains serializable IDs, decisions, observations, and measurements.
Images and masks remain in the case-scoped artifact store. SQLite checkpoints
are keyed by run ID; case records, run records, and append-only JSONL event logs
are stored on disk.

The HTTP API streams typed **NDJSON workflow events**. These are emitted from
graph node updates, not as model tokens or hidden chain-of-thought. A tool's
`tool_started` event currently arrives with that node's completed result, so it
is not an immediate start notification.

The console can restore its most recent case and run after refresh and resume a
run waiting for approval. Each new question starts a separate run with fresh
observations. Case artifacts persist and can inform later Qwen requests, but
cross-question conversational memory is not implemented. The workflow is
executed in the request stream, not by a background job queue.

## Retrieval and contour QC

Protocol lookup uses explicit aliases before a BM25 and optional embedding-based
ranking path. The bundled six-site registry is **research seed data**, with no
clinical citations. The guideline directory is empty except for instructions;
authorized Markdown or text documents must be added to enable passage retrieval.
Returned source labels are provenance aids, not evidence of clinical validation.

QC accepts one NIfTI mask per structure, using the filename as its anatomical
label. It requires the contour shape and affine to match the canonicalized case
grid; it does not resample mismatched contours. Checks include empty masks,
reference volume ranges, connected components, holes, laterality, and agreement
with VoxTell masks. VoxTell is a comparison reference, not ground truth. Surface
distance utilities exist, but the application leaves them disabled by default.

The deterministic `EvidenceCritic` checks whether numerical strings in an answer
occur in recorded evidence and warns about selected clinical terms without
protocol or QC observations. These heuristics can miss unsupported claims and
can flag legitimate derived values. They do not establish factual or clinical
correctness.

## API and interoperability

| Interface | Entry points |
| --- | --- |
| Case and artifact HTTP API | `POST /api/cases`, `GET /api/cases/{case_id}`, `POST /api/cases/{case_id}/contours`, `GET /api/cases/{case_id}/image`, `GET /api/cases/{case_id}/artifacts/{artifact_id}` |
| Run HTTP API | `POST /api/cases/{case_id}/runs`, `GET /api/runs/{run_id}`, `POST /api/runs/{run_id}/resume` |
| Health | `GET /api/health`; reports configuration, not model readiness. |
| MCP over stdio | `lookup_oar_protocol`, `segment_volume`, `run_contour_qc`; resource `segagent://cases/{case_id}`. |
| A2A research adapter | `GET /.well-known/agent-card.json`; `POST /a2a` supports `message/send` and `tasks/get`. |

The MCP segmentation and QC tools call domain tools directly and do not enter
the graph's approval gate. Graph QC also returns expert masks without the
segmentation-review interrupt. Artifact downloads are available independently
of approval. Review is therefore a workflow interaction, not an export or access
control boundary.

The A2A adapter requires an existing case ID and does not advertise streaming or
push notifications. It is not a certified A2A implementation.

## Research boundaries

The runtime has no authentication, tenant authorization, automatic
de-identification, or production upload controls. Case IDs separate local files;
they do not provide access control. Questions, answers, observations, filenames,
and imaging artifacts can contain sensitive content and are persisted locally.
OpenTelemetry spans currently use identifiers and counts; see
[configuration](configuration.md) for the unused trace-content setting.

The repository's deterministic tests and text-only rule-planner fixtures do not
validate segmentation accuracy, clinical performance, GPU requirements, or A2A
conformance. The software is intended for research with expert inspection of
results.
