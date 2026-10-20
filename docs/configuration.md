# Configuration

This page describes SegAgent Research v3. The source directory remains
`segagent-research-v2/` for path compatibility.

## Backend settings

`backend/config.py` defines Pydantic settings with the `SEGAGENT_` prefix.
Environment variables override a `.env` file read from the backend process's
**current working directory**. Start from `segagent-research-v2/` when using its
example configuration. Settings are cached per process; restart after changes.

In the table, `<project>` means the absolute `segagent-research-v2/` directory,
derived from the location of `backend/config.py`.

| Environment variable | Default | Purpose and limits |
| --- | --- | --- |
| `SEGAGENT_DATA_DIR` | `<project>/.research_data` | Root for case artifacts, conversations, messages, run metadata, and events. |
| `SEGAGENT_KNOWLEDGE_DIR` | `<project>/knowledge` | Protocol registry and optional guideline documents. |
| `SEGAGENT_CHECKPOINT_DB` | `<project>/.research_data/checkpoints.sqlite` | LangGraph SQLite checkpoints; independent of `DATA_DIR`. |
| `SEGAGENT_VOXTELL_MODEL` | Unset | Local VoxTell model directory; required for VoxTell-backed tools. |
| `SEGAGENT_LLM_MODEL` | `Qwen/Qwen3.5-9B` | Planner checkpoint ID or local path. Other models require compatible processor, template, and output behavior. |
| `SEGAGENT_ENABLE_THINKING` | `true` | Request the model's thinking channel and stream it separately. A template without that switch can ignore it. |
| `SEGAGENT_PLANNER` | `qwen` | `qwen` or deterministic `rule` planning. |
| `SEGAGENT_DEVICE` | `auto` | Backward-compatible fallback for both model stacks when their dedicated device settings are unset. |
| `SEGAGENT_LLM_DEVICE` | Unset | Planner loading device, falling back to `DEVICE`. `auto` selects CUDA if available, otherwise CPU. An indexed CUDA device pins the entire planner to that GPU. |
| `SEGAGENT_LLM_INPUT_DEVICE` | Unset | Device for request tensors and generation outputs. Otherwise inferred from loading and sharding settings. |
| `SEGAGENT_VOXTELL_DEVICE` | Unset | VoxTell device, falling back to `DEVICE`; `auto` selects CUDA if available, otherwise CPU. |
| `SEGAGENT_LLM_DEVICE_MAP` | `balanced_low_0` | Accelerate policy for automatic/bare-CUDA planner placement: `auto`, `balanced`, `balanced_low_0`, or `sequential`. |
| `SEGAGENT_LLM_MAX_MEMORY` | `{}` | JSON object of Accelerate memory budgets, such as `{"0":"18GiB","1":"18GiB","cpu":"64GiB"}`. Used with sharded CUDA placement. |
| `SEGAGENT_LLM_MAX_CONCURRENT_GENERATIONS` | `1` | Process-local limit for complete planner preparation/generation operations, from `1` to `4`. |
| `SEGAGENT_LLM_MAX_PROMPT_IMAGES` | `12` | Aggregate model-input image limit, from `3` to `64`; does not hide viewer masks. |
| `SEGAGENT_LLM_MAX_ARTIFACTS` | `2` | Maximum artifacts considered for planner mask-view context, from `0` to `16`. |
| `SEGAGENT_LLM_MASK_AXIAL_VIEWS` | `1` | Axial views per admitted mask, from `1` to `4`, subject to the aggregate image budget. |
| `SEGAGENT_LLM_MIN_IMAGE_TOKENS` | `256` | Per-image lower token budget, from `64` to `4096`; converted to pixels using processor patch and merge sizes. |
| `SEGAGENT_LLM_MAX_IMAGE_TOKENS` | `512` | Per-image upper token budget, from `64` to `4096`; must be at least `MIN_IMAGE_TOKENS`. |
| `SEGAGENT_VOXTELL_BATCH_SIZE` | `4` | Maximum prompts per internal VoxTell batch, from `1` to `32`. A large semantic request can span several batches. |
| `SEGAGENT_MAX_STEPS` | `8` | Planner step budget, from `2` to `24`; the workflow has a terminal fallback at its limit. |
| `SEGAGENT_MONTAGE_SLICES` | `3` | Axial scan grounding views, from `2` to `16`; coronal/sagittal views and the aggregate image budget also affect the final prompt. |
| `SEGAGENT_OVERLAY_SLICES` | `3` | Maximum saved axial overlays per nonempty segmentation mask, from `1` to `8`. |
| `SEGAGENT_MAX_NEW_TOKENS` | `1536` | Default generation budget for thinking and model output, from `256` to `8192`; some internal calls can override it. |
| `SEGAGENT_RETRIEVE_K` | `4` | Maximum guideline passage hits, from `1` to `20`. |
| `SEGAGENT_SEMANTIC_WEIGHT` | `0.45` | Embedding contribution to hybrid retrieval scores, from `0.0` to `1.0`; the remainder is lexical. Zero weight does not itself prevent embedding computation. |
| `SEGAGENT_TRACE_CONTENT` | `false` | Declared but not wired to tracing. It does not control local message/event persistence or thinking output. |
| `SEGAGENT_API_BASE_URL` | `http://127.0.0.1:8000` | URL advertised in the A2A Agent Card; does not set the bind address or frontend URL. |
| `SEGAGENT_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed browser origins. |

Both `Settings` and the checked-in `.env.example` now select `qwen` with
`Qwen/Qwen3.5-9B`. Set `SEGAGENT_PLANNER=rule` only for deterministic development
or evaluation. Rule planning does not replace VoxTell for segmentation, expert
QC, or semantic embeddings.

There is no mandatory mask-approval setting in the current runtime. Clarification
questions pause a turn and resume through the next message. Inspection and edits
are independent actions rather than an approval gate.

## Device placement and model input budgets

Use the dedicated LLM and VoxTell settings to avoid loading both stacks on the
same GPU unintentionally. `LLM_DEVICE=auto` or bare `cuda` enables the configured
device-map policy when CUDA is available. `LLM_DEVICE=cuda:1`, for example, pins
the planner to GPU 1 instead; its sharding and memory-map settings are then not
used for placement.

`LLM_MAX_MEMORY` is a JSON object. Numeric strings and `cuda:N` keys become CUDA
device indices. Omit a GPU from that map to reserve it for VoxTell. These are
model-placement budgets, not guarantees that request tensors, activations, and
KV caches fit within available memory. `LLM_INPUT_DEVICE` can explicitly choose
where request-local tensors live; with `balanced_low_0` and eligible GPU 0, the
default input device is `cuda:0`.

The included `.env.4x3090.example` puts the planner on GPUs 0–2 and VoxTell on GPU
3. It is a configuration example, not a minimum hardware requirement or a
performance benchmark. Keep one backend worker for a single local model service;
the generation semaphore and case locks are process-local, and extra workers
load additional model copies.

To reduce request memory, adjust prompt-image count, image-token limits,
artifact views, generation length, or VoxTell batch size. These controls limit
model inputs and internal batches; all stored masks can still be shown in the
viewer. No built-in quantization setting is provided.

Models load lazily. `/api/health` reports configuration and placement settings,
not successful model loading or inference readiness.

## Persistent paths

Changing `SEGAGENT_DATA_DIR` does **not** relocate the independently configured
checkpoint database. Set both when moving persistent state:

```dotenv
SEGAGENT_DATA_DIR=/absolute/path/to/segagent-data
SEGAGENT_CHECKPOINT_DB=/absolute/path/to/segagent-data/checkpoints.sqlite
```

Use absolute paths to avoid launch-directory dependence. The settings loader
creates the data directory and checkpoint parent. Preserve case artifacts,
conversation/message records, run logs, and checkpoints together when backing up
the workspace; stop the backend before making a consistent copy.

## Frontend settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Browser API origin for messages, cases, and artifacts. |

Set this separately, for example in `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the Vite development server or rebuild a static frontend after changing
it. The development port is `5173`; allow the browser's actual origin in
`SEGAGENT_CORS_ORIGINS`. `SEGAGENT_API_BASE_URL` does not override this setting.

## Traces, events, and research data

OpenTelemetry spans currently use identifiers, tool names, and counts. When the
OpenTelemetry stack initializes successfully, an OTLP HTTP exporter uses its
library environment configuration. `SEGAGENT_TRACE_CONTENT` is not currently an
active capture switch and must not be treated as a privacy control.

Conversation records, run events, streamed thinking, tool observations, review
or clarification text, filenames, and imaging artifacts can contain sensitive
content and persist locally. Thinking is model output, not admitted evidence.
The application does not automatically de-identify input. See
[architecture](architecture.md) for the model, dialogue, and verification limits.
