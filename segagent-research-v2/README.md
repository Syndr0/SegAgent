# SegAgent Research v3

This directory contains the **final, canonical SegAgent implementation**, merged from the `segagent-research-v3` development branch into `main`.

The directory name `segagent-research-v2/` is retained for path compatibility. It is a historical name, not a separate supported version.

![Research v3 architecture](../docs/assets/architecture.svg)

Start with the [project overview](../README.md) or [简体中文介绍](../README.zh-CN.md).

| Document | Purpose |
| --- | --- |
| [Getting started](../docs/getting-started.md) | Dependencies, weights, backend, frontend, and first conversation |
| [Architecture](../docs/architecture.md) | Persistent dialogue, Qwen3.5, VoxTell, QC, evidence, and integrations |
| [Configuration](../docs/configuration.md) | Device placement, input limits, batching, and storage |
| [Experiments](EXPERIMENTS.md) | Existing runners and proposed studies |
| [Contributing](../CONTRIBUTING.md) | Development workflow |
| [Security](../SECURITY.md) | Sensitive-data handling and research boundaries |

## Source layout

```text
backend/      Intent, dialogue, planner, tools, QC, evidence, storage, and API
frontend/     Case conversation, live model output, and NiiVue viewer
knowledge/    Research seed protocols and optional guideline documents
evals/        Rule fixtures and Qwen3.5 model-matrix evaluation
tests/        Deterministic unit and contract tests
```

Run backend and evaluation commands from this directory. Frontend commands run from `frontend/`. VoxTell code and model weights are external dependencies.

The default planner is `Qwen/Qwen3.5-9B` with thinking enabled. Masks appear without a blocking approval gate; clarification questions continue through the case conversation. Model thinking and mask comparisons do not replace clinical evidence or expert review.
