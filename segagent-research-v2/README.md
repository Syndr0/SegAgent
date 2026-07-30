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
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

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

- A named general organ goes directly to one VoxTell request.
- OAR contouring confirms or corrects the treatment site, retrieves its protocol
  once, and sends the complete OAR list in one batch.
- Tumor contouring requests spatial information when it is missing, handles an
  uncertain blind attempt explicitly, and only refines a first mask after a
  positive location confirmation.
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
