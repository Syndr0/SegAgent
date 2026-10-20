# Architecture

SegAgent Research v3 combines a local multimodal planner, text-prompted 3D
segmentation, persistent case conversations, and evidence-based contour analysis.
The application remains in `segagent-research-v2/` for path compatibility; that
directory name does not identify a separate product choice.

![SegAgent Research v3 architecture](assets/architecture.svg)

## Components

| Component | Implementation | Responsibility |
| --- | --- | --- |
| Research console | React, TypeScript, NiiVue | Case upload, conversational requests, image/mask inspection, and on-demand mask editing. |
| HTTP API | `backend/api.py` | Case and contour uploads, messages, conversation history, edited masks, and artifact downloads. |
| Planner | `backend/planner.py` | Local Qwen3.5 decisions, prompt normalization, visual answers, and incremental model-thinking output; a rule planner provides a deterministic baseline. |
| Dialogue and guards | `backend/intent.py`, `dialogue.py`, `guards.py` | Task routing, clarification stages, report prerequisites, artifact references, and repeated-request protection. |
| Workflow | `backend/workflow.py` | Execute typed actions, stream events, checkpoint conversations, and continue suspended clarification turns. |
| Segmentation adapter | `backend/expert.py`, `batching.py` | Lazily load VoxTell, bound segmentation prompt batches, and reuse its text embeddings for retrieval. |
| Domain tools | `backend/tools.py`, `qc.py` | Protocol lookup, segmentation, existing-mask measurement, ROI-aware QC, and report evidence. |
| Answer verification | `backend/verifier.py`, `tools.py` | Compare numerical claims and citations with admitted tool evidence and choose a fallback when checks fail. |
| Persistence | `backend/storage.py` | Case artifacts, provenance, conversations, messages, run records, and event logs; LangGraph uses SQLite checkpoints. |

## Models and image inputs

The default planner is **`Qwen/Qwen3.5-9B`**. The adapter uses Transformers Auto
model classes and an image/text processor, with BF16 on CUDA and FP32 on CPU.
Another checkpoint must be compatible with the adapter's processor, template,
and output contracts; changing a model ID alone is not a compatibility test.

Qwen receives representative axial slices, coronal and sagittal views, relevant
conversation history, current tool observations, and selected mask views. Image,
artifact, and per-image token budgets bound its prompt. It does not receive the
full 3D voxel array as a model input.

**VoxTell** processes the full canonicalized 3D volume. SegAgent normalizes source
prompts into anatomical concepts, pathological structures, or clinical
descriptions, preserving the original text and normalization provenance. Large
structure lists are processed in bounded internal batches. VoxTell also provides
the `embed_text_prompts` interface used for semantic retrieval; its text-encoding
stack is separate from the Qwen3.5 planner. VoxTell code and model assets are
external dependencies. This repository contains no training pipeline.

Generated masks and overlays are artifacts. Recorded measurements include voxel
count, volume, mean intensity, voxel bounding box, and voxel/world centroids.
Empty predictions remain explicit empty results. Generated tumor masks are
segmentation hypotheses, not diagnoses or validated clinical target volumes.

## Conversation and execution

Each case owns one durable conversation. `WorkflowService.send` starts a new
turn, or treats the message as a reply when that conversation has a pending
question. The LangGraph thread is keyed by conversation ID. Message history and
case artifacts remain available across turns, while observations, duplicate
signatures, decisions, and dialogue gates reset for each new turn. This prevents
the previous turn's measurements from being silently admitted as current tool
evidence.

The planner selects a validated action:

- `lookup_protocol`: retrieve a site protocol and optional guideline passages.
- `segment`: normalize prompts and run full-volume segmentation.
- `inspect_mask`: measure an existing contour or mask without rerunning VoxTell.
- `run_qc`: analyze uploaded contours by ROI role.
- `answer_question`: generate a descriptive visual answer.
- `ask_user`: suspend for clarification or a workflow-specific confirmation.
- `final`: finish using the observations collected for the turn.

Schema-guided JSON prompting and validation check decisions; this is not
constrained decoding. Deterministic dialogue stages and guards can ask for
missing site, tumor context, or report inputs before execution. Viewer context
includes referenced, visible, and selected artifact IDs, validated against the
case before use.

`ask_user` creates a LangGraph interrupt, a `question_pending` event, and a
`waiting_reply` run. The next message resumes that turn. **Segmentation does not
pause for mandatory mask approval.** It returns observations to planning, and
users can inspect or edit a mask independently of the run. Edited masks are
validated against the case grid and saved as new artifacts with derivation and
version metadata. Downloads have no approval gate.

The service allows one active response stream per case in its process. Qwen
generation has an additional configurable process-local concurrency limit. These
guards are not a distributed scheduler or a multi-worker job queue.

## Streaming and persistence

Qwen generation uses `TextIteratorStreamer`. Thinking-channel chunks are emitted
while generation runs as `thinking_delta` events through LangGraph custom
streaming and the HTTP NDJSON response. Phase, tool, observation, artifact, and
completion events share that stream. A completed decision or answer is still
validated and delivered as a complete event; the final answer is not a separate
token-by-token prose stream.

Model thinking is displayed separately from tool evidence and does not establish
a measurement or clinical fact. A template without a thinking switch may ignore
the setting. Reverse proxies can still affect delivery despite the API's
anti-buffering headers.

Serializable state lives in graph checkpoints. Uploaded scans, contours, masks,
and overlays live in case-scoped files; metadata includes artifact IDs and
hashes. Conversations, messages, run records, and append-only JSONL events are
persisted separately. The console can restore conversation history and continue
a pending question. The runtime executes work in the request stream rather than
a background task system; checkpointing is not a guarantee of arbitrary
crash recovery or inference cancellation.

## ROI-aware contour QC and reports

NIfTI masks must match the canonical case shape and affine. Mismatched masks are
not automatically resampled. QC classifies each uploaded ROI conservatively,
using explicit metadata when available, then recognized names:

| ROI role | Analysis |
| --- | --- |
| Organ/OAR | Compare with an independently generated VoxTell reference; record Dice, contour/model volume, signed volume difference, mean surface distance, HD95, reference masks, and difference overlays. |
| Target, including GTV/CTV/PTV | Describe volume, position, and distances to nearby contoured organs; mark `not_automatically_assessed`. No anatomy-model validation is applied. |
| Auxiliary planning structure | Inventory and measure as appropriate; exclude from anatomy-model comparison and clinical structure reports. |
| Unclassified | Preserve as unclassified and exclude from anatomy-model comparison until a role can be established. |

Surface metrics are **enabled by default** in the current QC engine. Model
disagreement flags are inspection aids; VoxTell is not ground truth. The current
engine does not use the older prototype's reference-volume, holes, components,
or laterality checks.

Protocol completeness is a conservative canonical-name set comparison, requiring
a supplied treatment site. Applicability records separate excluded or unresolved
sex-specific structures and identify research-seed sources. A missing structure
is not a judgement that a treatment plan is inadequate. The bundled six-site
registry is research seed data without clinical citations; the guideline folder
contains instructions, not a supplied clinical corpus.

Reports use deterministic readiness checks and artifact-bound measurements.
Comprehensive scope requires both organ/OAR and target evidence; available,
OAR-only, and target-only scopes have narrower prerequisites. Report candidate
selection prefers edited/derived masks over their predecessors. Auxiliary and
unclassified structures are excluded. A target description remains unassessed,
even when included in a report.

## Evidence verification and limits

`MetricEvidence` carries metric, value, unit, provenance, structure, and source
artifact IDs. `GroundingVerifier` compares answer numbers numerically with a
tolerance, checks recognized metric/unit context, and checks bracketed citations
against admitted retrieval sources. It does not search arbitrary serialized
metadata for matching digits. The critic can replace a failed answer with a
deterministic evidence summary or a number-free completion response.

These checks are heuristic text parsing, not semantic or clinical verification.
They cannot prove that a measurement is accurate, that a source is authoritative,
or that every claim refers to the correct structure. Descriptive visual answers
can have no tool evidence. Expert interpretation remains necessary.

## Interfaces

| Interface | Entry points |
| --- | --- |
| Cases | `POST /api/cases`, `GET /api/cases/{case_id}`, `POST /api/cases/{case_id}/contours` |
| Images and artifacts | `GET /api/cases/{case_id}/image`, `GET /api/cases/{case_id}/artifacts/{artifact_id}` |
| Edited masks | `POST /api/cases/{case_id}/edited-mask` |
| Conversations and runs | `POST /api/cases/{case_id}/messages`, `GET /api/cases/{case_id}/conversation`, `GET /api/runs/{run_id}` |
| Health | `GET /api/health`; configuration information, not model-readiness validation. |
| MCP over stdio | `lookup_oar_protocol`, `segment_volume`, `run_contour_qc`; resource `segagent://cases/{case_id}`. |
| A2A research adapter | `GET /.well-known/agent-card.json`; `POST /a2a` supports `message/send` and `tasks/get`. |

MCP tools call domain tools directly rather than running the whole dialogue.
`segment_volume` normalizes prompts and rejects generic pathology or mixed
anatomy/target requests that require additional context. `run_contour_qc` accepts
an optional site. The A2A adapter sends text through the conversation workflow
but returns a collected event result, not a streaming response. It requires an
existing case ID and is not a certified A2A implementation.

## Research boundaries

The application has no authentication, tenant authorization, automatic
de-identification, or production upload controls. Case scoping is not access
control. Local artifacts, messages, events, and checkpoints can contain sensitive
data, including streamed model thinking. OpenTelemetry currently records
identifiers and counts; the trace-content configuration field is not an active
content-capture switch. See [configuration](configuration.md).

Deterministic and fake-backend tests do not establish real-model segmentation
accuracy, clinical performance, hardware capacity, or interoperability
conformance. Legacy directory and metadata labels may retain historical version
strings; the documented application is SegAgent Research v3.
