<div align="center">

# SegAgent

**Conversational 3D segmentation. Typed tools. Evidence-grounded follow-up.**

A research framework connecting Qwen3.5 to VoxTell for medical-image segmentation, mask-aware dialogue, and contour quality control.

[![Status: Research](https://img.shields.io/badge/Status-Research-7c3aed?style=flat-square)](#project-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-2563eb?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)](segagent/pyproject.toml)
[![TypeScript 5.9](https://img.shields.io/badge/TypeScript-5.9-3178c6?style=flat-square&logo=typescript&logoColor=white)](segagent/frontend/package.json)

[![Planner: Qwen3.5-9B](https://img.shields.io/badge/Qwen3.5-9B-6366f1?style=flat-square)](docs/architecture.md#models-and-image-inputs)
[![Segmentation: VoxTell](https://img.shields.io/badge/VoxTell-3D-0891b2?style=flat-square)](docs/architecture.md#models-and-image-inputs)
[![LangGraph](https://img.shields.io/badge/LangGraph-1c3c3c?style=flat-square&logo=langgraph&logoColor=white)](docs/architecture.md#conversation-and-execution)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](segagent/backend/api.py)
[![React 19](https://img.shields.io/badge/React-19-149eca?style=flat-square&logo=react&logoColor=white)](segagent/frontend/package.json)
[![NiiVue](https://img.shields.io/badge/NiiVue-3D_viewer-475569?style=flat-square)](docs/architecture.md#components)

[简体中文](README.zh-CN.md) · [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) · [Contributing](CONTRIBUTING.md)

</div>

![SegAgent: persistent case conversation, Qwen3.5 planning, VoxTell segmentation, and role-aware QC](docs/assets/architecture.svg)

SegAgent separates planning from segmentation. **Qwen3.5-9B** interprets image slices, follows the case conversation, and selects structured actions; **VoxTell** segments the full 3D volume. Masks, overlays, measurements, conversation history, and run evidence remain associated with the case for follow-up questions and reports.

## Project status

The application is in **[`segagent/`](segagent/)**. The earlier plugin-based prototype is preserved under [`archive/prototype/`](archive/prototype/) for source history.

> **Research software.** SegAgent has not been clinically validated and is not a medical device. Model thinking and expert-model agreement are not clinical evidence. The included protocol registry is research seed data, not an approved clinical guideline set.

## Contents

- [Capabilities](#capabilities)
- [Quick start](#quick-start)
- [Conversation workflow](#conversation-workflow)
- [Repository map](#repository-map)
- [Evaluation](#evaluation)
- [Research boundaries](#research-boundaries)
- [Contributing and security](#contributing-and-security)
- [Acknowledgments and citation](#acknowledgments-and-citation)
- [License](#license)

## Capabilities

| Capability | Implementation |
| --- | --- |
| Persistent conversation | Case-bound message history, resumable clarification questions, and validated references to previously generated masks. |
| Structured planning | Typed intent/dialogue state and validated actions for protocol lookup, segmentation, inspection, QC, visual questions, clarification, and final responses. |
| Prompt normalization | Classify anatomical concepts, pathological structures, and clinical descriptions before sending validated prompts to VoxTell. |
| Volumetric segmentation | Full-volume VoxTell inference in bounded prompt chunks; NIfTI mask and PNG overlay artifacts with recorded measurements. |
| Mask-aware follow-up | Reopen case-owned masks, inspect geometry, and use overlays as context for visual questions and reports. |
| Role-aware contour QC | Distinguish organs, clinical targets, auxiliary ROIs, and unresolved labels; compare eligible organs with model references and attach typed metric evidence. |
| Live research console | React/NiiVue viewer with incrementally streamed model thinking, workflow progress, and generated masks. |
| Interoperability | MCP tools and an experimental A2A JSON-RPC adapter, reusing the same domain services. |

The models use pretrained weights; the repository does not include training or fine-tuning pipelines. See [Architecture](docs/architecture.md) for evidence checks and interface boundaries.

## Quick start

Requirements: **Python 3.11+**, compatible PyTorch/torchvision, **Transformers 5.7+ within the declared requirements**, **Node.js 22.12+**, and separately installed VoxTell code and weights. GPU memory depends on the combined models and input volume; no validated minimum is published. Follow the [complete setup guide](docs/getting-started.md) first.

From a prepared environment:

```bash
git clone https://github.com/Syndr0/SegAgent.git
cd SegAgent/segagent
cp .env.example .env
```

The example already selects Qwen3.5 with thinking enabled. Set your local model path in `.env`:

```dotenv
SEGAGENT_PLANNER=qwen
SEGAGENT_LLM_MODEL=Qwen/Qwen3.5-9B
SEGAGENT_ENABLE_THINKING=true
SEGAGENT_VOXTELL_MODEL=/absolute/path/to/voxtell_v1.1
```

Start the backend from the application directory:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, from the repository root:

```bash
cd segagent/frontend
npm install
npm run dev -- --host 127.0.0.1 --strictPort
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173), upload a NIfTI scan, and send a message. Models load on first use. Use one API worker and avoid `--reload` for GPU inference.

The [configuration reference](docs/configuration.md) covers independent Qwen/VoxTell placement, input budgets, batching, persistent paths, and `VITE_API_BASE_URL`. The [four-RTX-3090 profile](segagent/.env.4x3090.example) is a placement example, not a minimum requirement. The optional `rule` planner is a development baseline within the same application; segmentation still requires VoxTell.

## Conversation workflow

1. **Register a case.** Upload a `.nii` / `.nii.gz` scan and optional contours on the same grid.
2. **Send a request.** Intent and dialogue state determine whether to act or ask for missing information.
3. **Collect evidence.** Retrieve protocols, normalize prompts, segment, inspect masks, or build QC/report evidence.
4. **Continue in context.** Ask about a generated mask, answer a pending site/location question, or upload an edited mask.
5. **Inspect the result.** Read responses alongside measurements, artifacts, progress events, and evidence limitations.

Named organs and specific pathological structures can be sent directly for segmentation. Protocol-based OAR requests need a supplied or confirmed treatment site; explicitly named OARs can be segmented directly. Generic tumor requests ask for spatial information; uncertain attempts remain exploratory. Generated masks appear in the viewer without a mandatory approve/reject gate.

QC treats model comparisons as references, not ground truth. GTV/CTV/PTV and other clinical targets are measured and spatially described, with `not_automatically_assessed` status rather than an automatic declaration of correctness. Auxiliary and unresolved ROIs are reported separately. See [Architecture](docs/architecture.md) for applicability and evidence rules.

## Repository map

```text
SegAgent/
├── README.md / README.zh-CN.md   Project overview
├── LICENSE                     MIT license
├── CONTRIBUTING.md             Contribution workflow
├── SECURITY.md                 Security and data-handling guidance
├── docs/
│   ├── getting-started.md      Setup and first conversation
│   ├── architecture.md         Workflow, models, tools, and interfaces
│   ├── configuration.md        Environment-variable reference
│   └── assets/                 Architecture figure and editable source
├── segagent/                   SegAgent application
│   ├── backend/                Dialogue, planner, tools, QC, evidence, and APIs
│   ├── frontend/               Conversation UI and NiiVue viewer
│   ├── knowledge/              Seed protocols and guideline directory
│   ├── evals/                  Rule fixtures and model-matrix runner
│   └── tests/                  Deterministic unit and contract tests
└── archive/
    └── prototype/              Historical prototype source
```

## Evaluation

From an environment with the research dependencies installed:

```bash
cd segagent
pytest -q
python evals/run_evals.py --dataset evals/golden_cases.jsonl

# Model-backed evaluation on an already registered, de-identified case:
python evals/run_model_matrix.py --case-id case_... --thinking both
```

The rule-fixture runner checks text-based planning and retrieval. The model-matrix runner compares Qwen3.5-9B and 27B with thinking enabled/disabled through the actual multimodal planner; it is not an end-to-end segmentation or clinical benchmark. Unit tests cover conversation, dialogue gates, prompt normalization, evidence, QC, storage, and runtime guards without real model weights.

No benchmark results are claimed here. See the [evaluation protocol](segagent/EXPERIMENTS.md) for implemented metrics, comparison limits, and proposed studies.

## Research boundaries

The application is intended for local research with de-identified NIfTI data. It does not provide authentication, automatic de-identification, or a production deployment stack. Grounding checks and role classification reduce particular failure modes; they do not guarantee factual or clinical correctness. Human clinical review is required before any use outside the research application.

The current upload API accepts NIfTI. Historical DICOM/RTSTRUCT integration in the archived prototype is not the canonical application's interface.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) and use the [issue templates](https://github.com/Syndr0/SegAgent/issues/new/choose). Include your commit, configuration, and a minimal synthetic example. Keep medical images, identifying data, credentials, case stores, and model weights out of issues and pull requests.

For a suspected vulnerability, follow [SECURITY.md](SECURITY.md).

## Acknowledgments and citation

SegAgent builds on [VoxTell](https://github.com/MIC-DKFZ/VoxTell), [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B), [LangGraph](https://github.com/langchain-ai/langgraph), and [NiiVue](https://github.com/niivue/niivue). The historical prototype integrated with [voxtell-web-plugin](https://github.com/gomesgustavoo/voxtell-web-plugin).

When reporting work that uses SegAgent, record the repository commit, model revisions, configuration, and data split. Cite upstream models and tools according to their own guidance.

## License

Repository contributions are licensed under the [MIT License](LICENSE). External dependencies, pretrained weights, and upstream-derived components remain subject to their respective licenses.
