# Getting started with Research v2

[Project overview](../README.md) · [Configuration](configuration.md) · [Architecture](architecture.md)

This guide describes the standalone application in `segagent-research-v2/`. The original `segagent/` extension has a [separate integration guide](legacy.md). Model weights, medical images, and external dependency source code are not included in this repository.

## Requirements

- Python 3.11 or newer; Python 3.12 is a practical environment choice.
- [Node.js 22.12+](https://v7.vite.dev/guide/) and npm for the Vite 7 frontend.
- An installed VoxTell package and its `voxtell_v1.1` weights.
- A compatible PyTorch/torchvision pair. The checked-in Python requirements constrain Torch to `>=2.3,<2.9` and Transformers to `>=4.54,<5`.
- Enough system RAM, GPU memory, and disk for the planner, VoxTell, its text encoder, and volume-sized intermediate arrays. There is no validated minimum GPU configuration published for this implementation.

CUDA inference loads Qwen in BF16. Check your GPU's BF16 support before selecting it. Both models use the same `SEGAGENT_DEVICE` setting; this version has no separate per-model GPU placement or built-in quantization setting.

## 1. Clone and create an environment

```bash
git clone https://github.com/Syndr0/SegAgent.git
cd SegAgent/segagent-research-v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use your environment's Python executable in place of `python3.12` if needed, provided it satisfies the project requirement.

## 2. Install dependencies

Install a compatible GPU PyTorch build using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). For example, on Linux with an NVIDIA driver compatible with CUDA 12.6:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt 'voxtell==0.1.2'
python -m pip check
```

The Torch example is a dependency combination, not a hardware validation result. Select a different compatible wheel when required by your system. Install VoxTell and the project requirements together so the resolver can honor their overlapping Transformers constraints. VoxTell installation details are maintained in its [official repository](https://github.com/MIC-DKFZ/VoxTell).

If you develop VoxTell from a local checkout instead, install that checkout in the same environment. A separate `voxtell-web-plugin` checkout is not needed for Research v2.

## 3. Download the model components

The application expects a **local VoxTell model directory**. From the active environment:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

model_root = Path.home() / 'models' / 'segagent'
snapshot_download(
    'mrokuss/VoxTell',
    allow_patterns=['voxtell_v1.1/*'],
    local_dir=str(model_root),
)
print('SEGAGENT_VOXTELL_MODEL=' + str(model_root / 'voxtell_v1.1'))

# Optional pre-downloads: otherwise the relevant model loads on first use.
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
snapshot_download('Qwen/Qwen3-Embedding-4B')
PY
```

The VoxTell path must contain `plans.json` and `fold_0/checkpoint_final.pth`. Qwen3-Embedding-4B belongs to VoxTell's text-encoding stack; it is distinct from the Qwen2.5-VL planner. Downloads require network access to Hugging Face and sufficient disk space. Keep weights outside the Git repository.

## 4. Configure the application

```bash
cp .env.example .env
```

Edit `.env`, replacing the model path with the value printed above:

```dotenv
SEGAGENT_PLANNER=qwen
SEGAGENT_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
SEGAGENT_VOXTELL_MODEL=/absolute/path/to/voxtell_v1.1
SEGAGENT_DEVICE=auto
SEGAGENT_REQUIRE_MASK_APPROVAL=true
```

The example file initially selects `rule`. That provides a deterministic planning baseline, not fake segmentation; real segmentation, expert QC, and semantic embedding work still depend on VoxTell. Use `qwen` for the local multimodal planner.

The current planner adapter uses the Qwen2.5-VL model class. Treat a different model family as an implementation change rather than assuming that any vision-language checkpoint can be substituted by changing an ID.

Default case artifacts and SQLite checkpoints live in `.research_data/`. If moving them, set both `SEGAGENT_DATA_DIR` and `SEGAGENT_CHECKPOINT_DB`. Keep the backend stopped when copying a consistent backup. See the [full configuration reference](configuration.md).

## 5. Start the backend

From `segagent-research-v2/`, with the Python environment active:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Use one worker for local inference to avoid loading model copies in several processes. The app does not provide an authentication layer; the loopback binding is appropriate for local research.

API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `/api/health` reports service/configuration state; it does not load the models or prove inference readiness.

## 6. Start the frontend

In another terminal, from the repository root:

```bash
cd segagent-research-v2/frontend
npm install
npm run dev -- --host 127.0.0.1 --strictPort
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The frontend defaults to `http://127.0.0.1:8000` for API calls. If you change that address, set `VITE_API_BASE_URL` before starting or building the frontend, and update `SEGAGENT_CORS_ORIGINS` to match the frontend origin.

## 7. Work with a case

1. Choose **Upload scan** and select a de-identified `.nii` or `.nii.gz` image.
2. Optionally choose **Add contours** to attach binary NIfTI masks on the same grid. The QC engine expects matching shape and affine; it does not automatically resample them.
3. Submit a request for a visible structure, for example “Segment the liver.”
4. Inspect the returned mask and measurements in the viewer. When the graph requests review, approve, reject, or provide feedback.
5. Read the final response together with its run trace. Refreshing the same browser origin can restore the most recent case/run from the backend.

The planner initially sees representative 2D slices, not the full 3D image. VoxTell processes the volume. Every new question starts a new run; persistent artifacts and review/resume do not imply persistent chat memory.

## Optional interfaces

Run the MCP server from the research directory:

```bash
python -m backend.mcp_server
```

The FastAPI app also exposes `/.well-known/agent-card.json` and `/a2a`. These are experimental integrations. See [Architecture](architecture.md) for the available tools, routes, and review boundaries.

## Common setup issues

| Symptom | Check |
| --- | --- |
| Missing `voxtell` / `nnunetv2` | Install VoxTell in the same Python environment as the API. |
| Missing model files | Point `SEGAGENT_VOXTELL_MODEL` to the `voxtell_v1.1` subdirectory, not its parent. |
| Frontend loads but requests fail | Check `VITE_API_BASE_URL`, the API port, and CORS origins. |
| Rule mode still loads a model | Rule mode replaces the planner only; VoxTell-backed work still needs model weights. |
| GPU memory is exhausted | Check for duplicate backend processes and the combined model/volume workload. This version does not provide quantization or separate-device controls. |
| QC rejects a contour | Verify its shape and affine against the case image; no automatic resampling is performed. |
| The stream updates in chunks | The application emits node-level workflow events, not model-token updates; an additional HTTP proxy may also buffer them. |

Do not treat a successful service launch or unit test run as a real-model or clinical validation result.
