# SegAgent Research v2

SegAgent Research v2 is a stateful, evidence-first research framework for
agentic 3D medical-image segmentation and contour quality control. It keeps the
original idea—an LLM plans and VoxTell performs volumetric segmentation—but
rebuilds the runtime around typed tools, durable case state, explicit evidence,
visible model reasoning, deterministic multi-turn workflows, evaluation, and
interoperability.

The original `segagent/` tree is not modified. This directory is a separate
research implementation.

## Research questions

1. Does a typed, stateful agent select better segmentation tools than a
   free-form ReAct parser?
2. Does image-and-measurement feedback reduce unsupported final claims?
3. How much do protocol retrieval, deterministic dialogue state, mask-aware
   visual question answering, and an evidence critic improve reliability?
4. Can the same domain tools be reused through web, MCP, and A2A interfaces?

## Architecture

```text
case upload -> case/artifact store -> durable per-case conversation
                                      |-> typed intent/dialogue FSM
                                      |-> typed LLM prompt normalization
                                      |-> protocol retrieval
                                      |-> batched VoxTell segmentation
                                      |-> mask-conditioned VQA/inspection
                                      |-> deterministic QC/report evidence
                                      `-> evidence-checked final response

model tokens + node lifecycle -> typed NDJSON stream -> React agent conversation
domain tools      -> MCP server
workflow          -> A2A research adapter
```

Key rule: graph state contains JSON-serializable references and measurements,
not NumPy arrays or PIL images. Scan, contour, overlay, and mask files live in a
case-scoped artifact store and are addressed by opaque IDs.

## Layout

- `backend/`: typed models, storage, retrieval, planner, tools, workflow and API.
- `frontend/`: case-bound React/NiiVue research interface.
- `knowledge/`: auditable protocol data and guideline document area.
- `evals/`: golden trajectory dataset and evaluation runner.
- `tests/`: deterministic unit and contract tests that do not require model weights.

## Quick start

The code is designed to live beside a VoxTell checkout. This lightweight
research setup intentionally does not include production deployment files.

```bash
cd segagent-research-v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

export SEGAGENT_VOXTELL_MODEL=/path/to/voxtell_v1.1
export PYTHONPATH=/path/to/voxtell-web-plugin:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Keep exactly one API worker for local GPU inference. Do not add `--reload` or
start multiple Uvicorn/Gunicorn workers: each process can load its own Qwen and
VoxTell model copies and exhaust GPU memory. Generation is additionally limited
to one in-process request at a time.

In another terminal:

```bash
cd segagent-research-v2/frontend
npm install
npm run dev
```

In the app:

1. Choose **Upload scan**. The case is created as soon as the upload finishes.
2. Choose **Add contours** if you want to run QC on existing contours.
3. Ask a question or request a contour in the bottom composer.
4. Follow the live model-thinking and task-progress stream. Model reasoning is
   displayed separately and is explicitly not treated as clinical evidence.
5. Generated masks are added to the viewer as soon as VoxTell returns them.

The last case and run are restored after a page refresh when they are still
available on the local backend.

The first real agent run lazily loads the configured planner and VoxTell.
Set `SEGAGENT_PLANNER=rule` to exercise the workflow without an LLM, or
`SEGAGENT_PLANNER=qwen` for the local multimodal planner. The default interactive
checkpoint is `Qwen/Qwen3.5-9B` with thinking enabled; use
`SEGAGENT_LLM_MODEL=Qwen/Qwen3.5-27B` only after comparing it on the same local
case set.

### Four-GPU local profile

Copy the checked-in four-GPU profile, then set its VoxTell model path:

```bash
cp .env.4x3090.example .env
```

This profile uses the following bounded layout for four 24-GiB GPUs:

- `SEGAGENT_LLM_DEVICE=auto` and
  `SEGAGENT_LLM_DEVICE_MAP=balanced_low_0` let Accelerate place Qwen.
- `SEGAGENT_LLM_INPUT_DEVICE=cuda:0` keeps inputs and generation outputs on
  the GPU for which `balanced_low_0` reserves headroom.
- `SEGAGENT_LLM_MAX_MEMORY` caps GPUs 0-2 at `18GiB` each, allows CPU
  spillover up to `64GiB`, and excludes GPU 3 from Qwen placement.
- `SEGAGENT_VOXTELL_DEVICE=cuda:3` reserves GPU 3 for VoxTell's primary
  workload, while `SEGAGENT_VOXTELL_BATCH_SIZE=4` processes a large OAR set in
  bounded internal chunks.
- `SEGAGENT_LLM_MAX_CONCURRENT_GENERATIONS=1` prevents overlapping
  request-local activation and KV-cache peaks.
- Each Qwen request is capped at 12 prompt images and 2 artifact masks, with 1
  axial view per mask, 3 montage slices, and 256-512 visual tokens per image.
  Generation is capped at 1,536 new tokens.

These are model-input limits, not viewer controls. They do not hide or remove
masks from the scan viewer. `SEGAGENT_ENABLE_THINKING=true` remains the default,
so model thinking continues to stream live in its separate UI channel.

## Workflow

The planner returns a validated `PlannerDecision`, never an `ACTION:` string.
Supported actions are:

- `lookup_protocol`: retrieve an auditable OAR protocol and guideline passages.
- `segment`: run a typed, batched VoxTell segmentation request.
- `inspect_mask`: re-open and measure an existing case-owned mask.
- `run_qc`: compare uploaded contours with deterministic checks and an expert mask.
- `answer_question`: perform mask-aware visual question answering.
- `ask_user`: suspend with a typed clarification or confirmation.
- `final`: synthesize a response grounded in collected observations.

The workflow itself enforces the multi-turn sequences:

- Before segmentation, the existing Qwen instance classifies each prompt as an
  `anatomical_concept`, `pathological_structure`, or `clinical_description`.
  Anatomical aliases such as `eye_L` can become `left eye`; validated pathology
  phrases and report descriptions retain their original wording.
  For example, `iver` becomes the anatomical prompt `liver`, `bladder
  carcinoma` remains a pathological-structure prompt, and `Bronchial carcinoma
  in the right upper lobe with pleural contact` remains one complete clinical
  description.
- A named general organ or specific pathological structure goes directly to one
  VoxTell request. A generic bare `tumor`, `mass`, or `lesion` still triggers the
  spatial-description dialogue.
- OAR contouring confirms or corrects the treatment site, retrieves its protocol
  once, and sends the complete OAR list in one batch.
- Tumor contouring requests spatial information when it is missing, handles an
  uncertain blind attempt explicitly, and only refines a first mask after a
  positive location confirmation. If the user cannot supply any location, the
  blind attempt sends the report-style prompt `carcinoma` to VoxTell while the
  viewer label remains the user's original `tumor`/`mass`/`lesion` wording; the
  resulting mask is marked exploratory and does not imply a diagnosis.
- Follow-up questions receive only validated artifacts belonging to the current
  case and can use mask overlays as visual context.
- QC compares non-target contours with a second model. Clinical target volumes
  are marked `not_automatically_assessed`, measured and spatially described, but
  never declared correct.

Masks do not pass through a blocking approval gate. A clinician may still upload
an edited version, and every target remains labelled for mandatory clinical
review before any use outside this research application.

## Evaluation

```bash
pytest -q
# Or, without pytest:
python -m unittest discover -s tests -v
python evals/run_evals.py --dataset evals/golden_cases.jsonl

# On a de-identified case already uploaded to the local store:
python evals/run_model_matrix.py --case-id case_... --thinking both
```

The starter harness measures action selection, requested-structure recall,
protocol-site selection, and abstention behavior. Extend it with annotated
volumes and final-answer labels to measure Dice, HD95, latency, GPU memory,
and unsupported-claim rate. The model-matrix runner evaluates 9B and 27B with
thinking on and off using the actual multimodal planner; it also records schema
validity, thinking availability, and latency. See `EXPERIMENTS.md` for the full
reporting protocol.

## MCP and A2A

Run the optional MCP server:

```bash
python -m backend.mcp_server
```

The FastAPI app publishes an A2A Agent Card and a small JSON-RPC research
adapter. It is useful for interoperability experiments but is not presented as
a certified A2A implementation; conformance tests should be added before any
production claim.

## Safety boundary

This is research software, not a medical device. It must not autonomously make
clinical decisions. Keep patient data local, disable trace-content capture by
default, validate image geometry, treat displayed thinking as model output rather
than evidence, require expert review before export, and cite the exact approved
protocol source used for any clinical statement. The protocol JSON committed here
is research seed data and is not an institution-approved clinical protocol.
