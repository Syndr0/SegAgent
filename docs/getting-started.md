# Getting started with SegAgent

[Project overview](../README.md) · [Configuration](configuration.md) · [Architecture](architecture.md)

The SegAgent application lives in `segagent/`. Follow the steps below to install
its dependencies, configure the models, and start a local research session.

## Requirements

- Python 3.11+, with Python 3.12 a practical environment choice.
- [Node.js 22.12+](https://v7.vite.dev/guide/) and npm for Vite 7.
- Separately installed VoxTell code and `voxtell_v1.1` weights.
- A compatible PyTorch/torchvision pair. The project requires Torch `>=2.3,<2.9`, Transformers `>=5.7,<6`, and `qwen-vl-utils>=0.0.14`.
- RAM, GPU memory, and disk for Qwen3.5, VoxTell, its text encoder, and volume-sized arrays. No validated minimum configuration is published.

The CUDA planner uses BF16. Placement and input budgets can be configured independently for the planner and segmentation model; see [Configuration](configuration.md).

## 1. Clone and create an environment

```bash
git clone https://github.com/Syndr0/SegAgent.git
cd SegAgent/segagent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use your Python executable in place of `python3.12` when necessary, provided it meets the project requirement.

## 2. Install dependencies

Select a compatible build using the [official PyTorch table](https://pytorch.org/get-started/previous-versions/). For example, on Linux with a driver compatible with CUDA 12.6:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt 'voxtell==0.1.2'
python -m pip check
```

This is a dependency combination, not a hardware validation claim. Install VoxTell and project requirements together to resolve their constraints jointly. A local VoxTell checkout may also be installed in the same environment; a separate web-plugin checkout is not needed.

## 3. Download model components

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

# Optional pre-downloads; otherwise fetched on first relevant use.
snapshot_download('Qwen/Qwen3.5-9B')
snapshot_download('Qwen/Qwen3-Embedding-4B')
PY
```

The local VoxTell directory must contain `plans.json` and `fold_0/checkpoint_final.pth`. Qwen3-Embedding-4B is VoxTell's text encoder, distinct from the Qwen3.5 planner. Keep weights outside Git and record model revisions for experiments.

## 4. Configure the application

```bash
cp .env.example .env
```

The example already selects the multimodal planner and thinking. Replace the model path:

```dotenv
SEGAGENT_PLANNER=qwen
SEGAGENT_LLM_MODEL=Qwen/Qwen3.5-9B
SEGAGENT_ENABLE_THINKING=true
SEGAGENT_VOXTELL_MODEL=/absolute/path/to/voxtell_v1.1
```

For hardware matching the four-3090 profile, you can use `.env.4x3090.example` instead. It reserves one GPU for VoxTell and applies a Qwen placement budget to the other GPUs. It is an example, not a requirement for four GPUs or an OOM guarantee.

For other layouts, configure the LLM device, input device, memory map, and VoxTell device explicitly. Indexed Qwen placement such as `cuda:0` pins the whole model and bypasses the automatic placement memory map. Input limits constrain Qwen prompts, not masks visible in the viewer.

The optional `rule` planner is a development baseline in the same application; segmentation, expert QC, and semantic retrieval still require VoxTell.

## 5. Start the backend

From the application directory, with the Python environment active:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep one worker and avoid `--reload`: extra processes can load duplicate models. Generation concurrency is limited within the process, and the conversation service prevents overlapping active turns for the same case.

API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). `/api/health` reports configuration, not successful model loading. There is no application authentication layer.

## 6. Start the frontend

In another terminal, from the repository root:

```bash
cd segagent/frontend
npm install
npm run dev -- --host 127.0.0.1 --strictPort
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). If the backend address changes, set `VITE_API_BASE_URL` before starting/building the frontend and configure `SEGAGENT_CORS_ORIGINS` for the browser origin.

## 7. Work through a conversation

1. Upload a de-identified `.nii` / `.nii.gz` scan.
2. Optionally add binary NIfTI contours on the same grid for QC.
3. Ask for a named structure, such as “Segment the liver.”
4. Inspect the mask when it appears, then ask about it in the same conversation.
5. Answer pending site/location questions in the composer. Every mask does not require approval.
6. Upload an edited mask or request QC/reporting. Targets are not automatically declared correct.

Thinking appears separately from evidence and may contain conjecture. The viewer displays available masks independently of Qwen's prompt image budget.

Messages, artifacts, and checkpoints persist locally. The same browser origin can restore the previous case and conversation when backend data remains available. Set both `SEGAGENT_DATA_DIR` and `SEGAGENT_CHECKPOINT_DB` when relocating state; stop the backend before taking a consistent backup.

## Optional interfaces

```bash
python -m backend.mcp_server
```

The FastAPI app also exposes `/.well-known/agent-card.json` and `/a2a`. See [Architecture](architecture.md) for interfaces and limitations.

## Common issues

| Symptom | Check |
| --- | --- |
| Missing VoxTell or nnU-Net imports | Install VoxTell in the API's Python environment. |
| Missing weights | Point to `voxtell_v1.1`, not its parent. |
| Qwen3.5 cannot load | Check the actual Transformers version and project constraints. |
| API requests fail | Check `VITE_API_BASE_URL`, binding, and CORS. |
| GPU OOM | Check duplicate workers, placement, image/token budgets, and VoxTell batch size. Restart after correcting placement. |
| QC rejects geometry | Match the canonical case grid; contours are not automatically resampled. |
| Thinking arrives late | Check HTTP proxy buffering; model deltas and events share NDJSON. |
| Data disappears after relocation | Check both artifact storage and checkpoint paths. |

A successful launch or deterministic unit test does not establish model accuracy or clinical validity.
