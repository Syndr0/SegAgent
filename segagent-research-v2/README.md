# SegAgent Research v2

SegAgent Research v2 is a stateful, evidence-first research framework for
agentic 3D medical-image segmentation and contour quality control. It keeps the
original idea—an LLM plans and VoxTell performs volumetric segmentation—but
rebuilds the runtime around typed tools, durable case state, explicit evidence,
human review, evaluation, and interoperability.

The original `segagent/` tree is not modified. This directory is a separate
research implementation.

## Research questions

1. Does a typed, stateful agent select better segmentation tools than a
   free-form ReAct parser?
2. Does image-and-measurement feedback reduce unsupported final claims?
3. How much do protocol retrieval, an evidence critic, and human review improve
   contour-workflow reliability?
4. Can the same domain tools be reused through web, MCP, and A2A interfaces?

## Architecture

```text
case upload -> case/artifact store -> durable LangGraph workflow
                                      |-> protocol retrieval
                                      |-> VoxTell segmentation tool
                                      |-> deterministic QC/evidence critic
                                      |-> human approval interrupt
                                      `-> cited final response

all node updates -> typed NDJSON events -> React research console
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
3. Enter a request, or use an example, and choose **Run**.
4. When **Review results** appears, inspect each mask in the viewer. Approve it,
   request changes with a note, or reject it.

The last case and run are restored after a page refresh when they are still
available on the local backend.

The first real agent run lazily loads the configured planner and VoxTell.
Set `SEGAGENT_PLANNER=rule` to exercise the workflow without an LLM, or
`SEGAGENT_PLANNER=qwen` for the local multimodal planner.

## Workflow

The planner returns a validated `PlannerDecision`, never an `ACTION:` string.
Supported actions are:

- `lookup_protocol`: retrieve an auditable OAR protocol and guideline passages.
- `segment`: run a typed, batched VoxTell segmentation request.
- `run_qc`: compare uploaded contours with deterministic checks and an expert mask.
- `ask_user`: stop with a precise clarification request.
- `final`: synthesize a response grounded in collected observations.

Segmentation artifacts pass through a LangGraph interrupt. The reviewer can
approve, reject, or attach feedback; the checkpoint then resumes at the same
case/run state.

## Evaluation

```bash
pytest -q
# Or, without pytest:
python -m unittest discover -s tests -v
python evals/run_evals.py --dataset evals/golden_cases.jsonl
```

The starter harness measures action selection, requested-structure recall,
protocol-site selection, and abstention behavior. Extend it with annotated
volumes and final-answer labels to measure Dice, HD95, latency, GPU memory,
reviewer-overturn rate, and unsupported-claim rate. See `EXPERIMENTS.md` for the
baseline/ablation matrix and reporting protocol.

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
default, validate image geometry, require expert review before export, and cite
the exact protocol source used for any clinical statement.
