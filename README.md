<div align="center">

# SegAgent

**Language-guided 3D segmentation. Typed tools. Reviewable evidence.**

A research framework connecting a local multimodal planner to VoxTell for medical-image segmentation and contour quality control.

[![License: MIT](https://img.shields.io/badge/License-MIT-2563eb.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-334155.svg)](segagent-research-v2/pyproject.toml)
[![Status: Research](https://img.shields.io/badge/Status-Research-0f766e.svg)](#research-status)

[简体中文](README.zh-CN.md) · [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) · [Contributing](CONTRIBUTING.md)

</div>

![SegAgent Research v2 architecture: case input, typed planning, VoxTell tools, evidence, review, and persistent artifacts](docs/assets/architecture.svg)

SegAgent separates planning from segmentation. **Qwen2.5-VL-7B-Instruct** interprets representative image slices and selects a structured action; **VoxTell** segments the full 3D volume. Measurements, overlays, tool observations, and review decisions are recorded alongside the case so a run can be inspected and resumed.

The repository contains two implementations. **Start with [`segagent-research-v2/`](segagent-research-v2/)** for the standalone application. [`segagent/`](segagent/) preserves the original extension for an external VoxTell web plugin.

> **Research software.** SegAgent has not been clinically validated and is not a medical device. Generated masks and QC findings require expert interpretation. The included protocol registry is research seed data, not an approved clinical guideline set.

## Contents

- [Capabilities](#capabilities)
- [Choose an implementation](#choose-an-implementation)
- [Quick start](#quick-start)
- [Workflow](#workflow)
- [Repository map](#repository-map)
- [Evaluation](#evaluation)
- [Research status](#research-status)
- [Contributing and security](#contributing-and-security)
- [Acknowledgments and citation](#acknowledgments-and-citation)
- [License](#license)

## Capabilities

| Capability | Research v2 implementation |
| --- | --- |
| Structured planning | Pydantic-validated `lookup_protocol`, `segment`, `run_qc`, `ask_user`, and `final` decisions. |
| Volumetric segmentation | Text prompts sent to VoxTell; NIfTI mask and PNG overlay artifacts, with recorded volume, bounding-box, and centroid measurements. |
| Protocol retrieval | Local OAR protocol registry plus lexical/embedding retrieval over user-supplied Markdown and text documents. |
| Contour QC | Geometry checks, empty-mask and topology checks, reference-volume checks, laterality checks, and Dice agreement with an expert-model mask. |
| Human review | Graph segmentation pauses for approval, rejection, or feedback, then resumes from its checkpoint. |
| Inspectable runs | React/NiiVue viewer, typed NDJSON workflow events, file-backed case/run records, and SQLite graph checkpoints. |
| Tool integration | Three MCP tools and an experimental A2A JSON-RPC adapter. |

The planner and segmentation expert use pretrained weights; this repository does not include training or fine-tuning pipelines. The web stream reports workflow events rather than token-by-token model reasoning. See [architecture and implementation boundaries](docs/architecture.md).

## Choose an implementation

| | Research v2 | Original plugin extension |
| --- | --- | --- |
| Directory | [`segagent-research-v2/`](segagent-research-v2/) | [`segagent/`](segagent/) |
| Application | Standalone FastAPI backend and React/NiiVue frontend | Files copied into an external `voxtell-web-plugin` checkout |
| Planner | Typed JSON decisions | Free-form ReAct action parsing |
| State | Case artifacts, run records, checkpoint/resume | Prototype-specific state |
| Review | Checkpointed segmentation review | Earlier prototype behavior |
| Documentation | [Getting started](docs/getting-started.md) | [Legacy integration](docs/legacy.md) |

Both implementations depend on externally installed VoxTell code and model weights. Their configuration names and API routes differ; do not mix installation instructions between them.

## Quick start

Research v2 requires **Python 3.11+**, a compatible PyTorch environment, **Node.js 22.12+** for the frontend, and VoxTell weights. GPU memory requirements depend on image size and the combined model workload; this repository does not publish a validated minimum. Follow the [complete setup guide](docs/getting-started.md) to install the dependencies and download weights.

From a prepared environment:

```bash
git clone https://github.com/Syndr0/SegAgent.git
cd SegAgent/segagent-research-v2
cp .env.example .env
```

Set the model path and enable the multimodal planner in `.env`:

```dotenv
SEGAGENT_PLANNER=qwen
SEGAGENT_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
SEGAGENT_VOXTELL_MODEL=/absolute/path/to/voxtell_v1.1
SEGAGENT_REQUIRE_MASK_APPROVAL=true
```

Start the backend from `segagent-research-v2/`:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, from the repository root:

```bash
cd segagent-research-v2/frontend
npm install
npm run dev -- --host 127.0.0.1 --strictPort
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173), upload a NIfTI scan, and submit a request. Models load on first use. The checked-in `.env.example` starts with the **rule baseline**; switching to `qwen` enables the local multimodal planner. Rule mode still needs VoxTell for segmentation and expert QC.

For separate frontend/backend hosts, set `VITE_API_BASE_URL` and matching backend CORS origins. Configuration, persistent paths, and optional integrations are documented in [Configuration](docs/configuration.md).

## Workflow

1. **Register a case.** Upload a `.nii` / `.nii.gz` scan and optional masks on the same grid.
2. **Plan an action.** Qwen receives representative 2D slices, tool observations, and a JSON schema; the workflow validates its decision.
3. **Collect evidence.** Retrieve a protocol, run full-volume segmentation, or compare uploaded contours with deterministic checks and a model reference.
4. **Review segmentation.** The graph pauses after producing masks. Inspect overlays and approve, reject, or supply feedback to resume the run.
5. **Inspect the result.** Read the final response together with its measurements, observations, artifacts, and event trace.

The review interrupt covers graph-generated segmentation. MCP direct tool calls and QC references do not use that same gate, and artifact downloads are not approval-gated. Case artifacts persist across runs; a new question starts a new run rather than a conversation with persistent chat memory.

## Repository map

```text
SegAgent/
├── README.md / README.zh-CN.md   Project overview
├── LICENSE                     MIT license
├── CONTRIBUTING.md             Contribution workflow
├── SECURITY.md                 Security and data-handling guidance
├── docs/
│   ├── getting-started.md      Research v2 setup and first workflow
│   ├── architecture.md         Components, interfaces, and boundaries
│   ├── configuration.md        Environment-variable reference
│   ├── legacy.md               Original plugin integration
│   └── assets/                 Architecture figure and editable source
├── segagent-research-v2/
│   ├── backend/                Planner, tools, graph, storage, and APIs
│   ├── frontend/               React/NiiVue research console
│   ├── knowledge/              Seed protocols and guideline directory
│   ├── evals/                  Rule-planner fixtures and evaluation runner
│   └── tests/                  Deterministic unit tests
└── segagent/                   Original web-plugin extension
```

## Evaluation

From an environment with the research dependencies installed:

```bash
cd segagent-research-v2
pytest -q
python evals/run_evals.py --dataset evals/golden_cases.jsonl
```

The unit tests cover schemas, rule planning, retrieval, and storage. The starter evaluation uses text-only rule-planner fixtures. Neither establishes real-model segmentation accuracy, clinical quality, hardware capacity, or Qwen performance. No measured benchmark results are published here. See the [experiment protocol](segagent-research-v2/EXPERIMENTS.md) for proposed comparisons and reporting requirements.

## Research status

The current application is intended for local research with de-identified NIfTI data. It does not provide authentication, automatic de-identification, or a production deployment stack. Its heuristic evidence critic and expert-model agreement are aids to inspection, not correctness guarantees. Surface-distance metrics are not enabled in the default QC service.

DICOM conversion and RTSTRUCT-related code belongs to the original plugin integration; the Research v2 upload API accepts NIfTI. See [Security](SECURITY.md) and [Architecture](docs/architecture.md) for the current boundaries.

## Contributing and security

Bug reports, documentation improvements, and reproducible research contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), use the [issue templates](https://github.com/Syndr0/SegAgent/issues/new/choose), and include the implementation, configuration, and commit you used. Keep medical images, identifying data, credentials, and model weights out of issues and pull requests.

For a suspected vulnerability, follow [SECURITY.md](SECURITY.md) rather than publishing sensitive details in a public issue.

## Acknowledgments and citation

SegAgent builds on [VoxTell](https://github.com/MIC-DKFZ/VoxTell), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [LangGraph](https://github.com/langchain-ai/langgraph), and [NiiVue](https://github.com/niivue/niivue). The original extension integrates with [voxtell-web-plugin](https://github.com/gomesgustavoo/voxtell-web-plugin).

When reporting work that uses SegAgent, identify this repository and the exact commit, model checkpoints, configuration, and data split. Cite the upstream models and tools according to their own citation guidance. There is no SegAgent paper or DOI declared in this repository.

## License

Repository contributions are licensed under the [MIT License](LICENSE). External dependencies, pretrained weights, and upstream-derived components remain subject to their respective licenses; the MIT license does not grant additional rights to those materials.
