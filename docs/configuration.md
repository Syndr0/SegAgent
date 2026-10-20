# Configuration

This page covers `segagent-research-v2/`. Its settings are separate from the
environment variables used by the legacy `segagent/` overlay.

## Backend settings

`backend/config.py` defines Pydantic settings with the `SEGAGENT_` prefix.
Environment variables override values loaded from a `.env` file in the backend
process's **current working directory**. Start the backend from
`segagent-research-v2/` when using its `.env` file. Settings are cached per process;
restart the backend after changing them.

In the table, `<project>` means the absolute `segagent-research-v2/` directory,
derived from the location of `backend/config.py`.

| Environment variable | Default | Purpose and limits |
| --- | --- | --- |
| `SEGAGENT_DATA_DIR` | `<project>/.research_data` | Root directory for case files, artifacts, run metadata, and event logs. |
| `SEGAGENT_KNOWLEDGE_DIR` | `<project>/knowledge` | Directory containing `protocols.json`, `organ_reference.json`, and optional guideline documents. |
| `SEGAGENT_CHECKPOINT_DB` | `<project>/.research_data/checkpoints.sqlite` | SQLite database used by the LangGraph checkpointer. Independent of `DATA_DIR`. |
| `SEGAGENT_VOXTELL_MODEL` | Unset | Local VoxTell model directory. Required when a tool loads VoxTell. |
| `SEGAGENT_LLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Planner model ID or local path. The adapter uses the Qwen2.5-VL model class; this is not a general model-provider interface. |
| `SEGAGENT_PLANNER` | `qwen` | `qwen` for the local multimodal planner, or `rule` for deterministic planning. |
| `SEGAGENT_DEVICE` | `auto` | `auto` selects CUDA when available, otherwise CPU. Explicit PyTorch device strings are accepted by the model adapters; no CPU performance guarantee is implied. |
| `SEGAGENT_MAX_STEPS` | `8` | Planner step budget, from `2` to `24`. A subsequent limit check returns a final evidence summary. |
| `SEGAGENT_MONTAGE_SLICES` | `6` | Number of axial grounding views, from `2` to `16`. Coronal and sagittal mid-plane views are added separately. |
| `SEGAGENT_OVERLAY_SLICES` | `3` | Maximum axial overlay images generated per nonempty segmentation mask, from `1` to `8`. |
| `SEGAGENT_MAX_NEW_TOKENS` | `512` | Maximum new tokens for each Qwen decision generation, from `64` to `4096`. |
| `SEGAGENT_REQUIRE_MASK_APPROVAL` | `true` | Pause the graph after segmentation generates masks. Does not gate QC, direct MCP tools, or artifact downloads. |
| `SEGAGENT_RETRIEVE_K` | `4` | Maximum retrieved guideline passages, from `1` to `20`. |
| `SEGAGENT_SEMANTIC_WEIGHT` | `0.45` | Embedding contribution to hybrid retrieval scores, from `0.0` to `1.0`; the remainder is lexical. Setting this to zero does not prevent the current retrieval code from computing embeddings. |
| `SEGAGENT_TRACE_CONTENT` | `false` | Declared setting only. It is not wired to the tracing implementation and currently does not enable or disable content capture. |
| `SEGAGENT_API_BASE_URL` | `http://127.0.0.1:8000` | Public URL advertised by the A2A Agent Card. Does not configure the server bind address or the frontend API URL. |
| `SEGAGENT_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated browser origins allowed by FastAPI CORS middleware. |

The checked-in `.env.example` selects **`SEGAGENT_PLANNER=rule`**, although the
settings class defaults to **`qwen`** when no override is supplied. Copying the
example therefore starts with the deterministic planner. Set it to `qwen` to use
the configured local multimodal model. The rule planner still needs VoxTell for
segmentation and QC; it is not a synthetic segmentation backend.

Both model adapters load lazily. A successful `/api/health` response and
`voxtellConfigured: true` indicate configuration, not successful model loading or
inference.

## Storage paths

`DATA_DIR` and `CHECKPOINT_DB` have separate defaults. Changing only
`SEGAGENT_DATA_DIR` does **not** move the checkpoint database. Set both values
when relocating all persistent run state:

```dotenv
SEGAGENT_DATA_DIR=/absolute/path/to/segagent-data
SEGAGENT_CHECKPOINT_DB=/absolute/path/to/segagent-data/checkpoints.sqlite
```

Use absolute paths to avoid dependence on the launch directory. The settings
loader creates the data directory and the checkpoint database's parent
directory. Keep artifacts, run logs, and checkpoints together when backing up a
research workspace; checkpoints refer to case artifacts by ID.

## Frontend settings

The Vite frontend uses a separate variable:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend origin used for browser API and artifact requests. |

Set it in the frontend environment, for example in `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the Vite development server after changing it. For a built frontend,
rebuild to embed the new value. Vite uses port `5173` for development; the backend
must allow the browser's actual origin in `SEGAGENT_CORS_ORIGINS`.
`SEGAGENT_API_BASE_URL` does not override this frontend setting.

## Traces, events, and review

The OpenTelemetry facade creates spans with identifiers, tool names, and counts.
When the installed OpenTelemetry stack initializes successfully, it configures an
OTLP HTTP exporter using that library's environment configuration. The
`SEGAGENT_TRACE_CONTENT` field is currently unused; do not rely on it as a privacy
control.

Separately, local run event logs persist the question, answer, tool observations,
and review feedback. Uploaded images, masks, overlays, and filenames also remain
in the case store. The application does not automatically de-identify them.

Keep `SEGAGENT_REQUIRE_MASK_APPROVAL=true` for interactive segmentation review.
This setting controls the graph's review interrupt; it is not an authorization
policy or an export restriction. See [architecture](architecture.md) for the
paths that bypass that review node.
