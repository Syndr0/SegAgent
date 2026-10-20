# SegAgent Research v2

The standalone research application for language-guided 3D segmentation and contour QC: FastAPI, React/NiiVue, a Qwen2.5-VL planner, VoxTell tools, and a checkpointed LangGraph workflow.

![Research v2 architecture](../docs/assets/architecture.svg)

Start with the [project overview](../README.md) or [简体中文介绍](../README.zh-CN.md). Setup and implementation documentation live in one place:

| Document | Purpose |
| --- | --- |
| [Getting started](../docs/getting-started.md) | Requirements, model weights, backend, frontend, and first case |
| [Architecture](../docs/architecture.md) | Planning, tools, review, persistence, HTTP, MCP, and A2A |
| [Configuration](../docs/configuration.md) | Environment variables and defaults |
| [Experiments](EXPERIMENTS.md) | Evaluation design and reporting protocol |
| [Contributing](../CONTRIBUTING.md) | Development and contribution workflow |
| [Security](../SECURITY.md) | Research use and sensitive-data handling |

## Source layout

```text
backend/      Typed contracts, model adapters, tools, graph, storage, and API
frontend/     Case viewer, run events, and segmentation review
knowledge/    Research seed protocols, reference volumes, and guideline directory
evals/        Text-only rule-planner fixtures and runner
tests/        Deterministic schema, planner, retrieval, and storage tests
```

Run backend and evaluation commands from this directory so relative `.env` and dataset paths resolve correctly. Frontend commands run from `frontend/`.

VoxTell code and weights are external dependencies. The example configuration selects the rule baseline; actual segmentation still requires VoxTell. This is research software, with no clinical validation or production security claim.
