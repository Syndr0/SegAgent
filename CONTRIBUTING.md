# Contributing to SegAgent

SegAgent is an application for agentic 3D
medical-image segmentation, case-bound dialogue, and contour quality control.
Contributions to code, documentation, evaluation, and reproducible research
workflows are welcome.

The application source is in `segagent/`. The earlier plugin-based prototype
is preserved under `archive/prototype/` for source history; new development
targets the application.

## Discuss the change

Search existing [issues](https://github.com/Syndr0/SegAgent/issues) before opening
a new one. For a substantial feature, architecture change, or new dependency,
open an issue to discuss the intended behavior and scope before implementation.
Small documentation corrections can go directly to a pull request.

Identify the affected component, such as planning, dialogue, retrieval,
segmentation, evidence/QC, the frontend, or documentation. Check current behavior
against the code on `main`, rather than inferring it from archived modules.

## Set up a development environment

Use Python 3.11 or newer. From the repository root:

```bash
cd segagent
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Keep local paths and secrets in `.env`. Model weights and external VoxTell
checkouts are separate dependencies; their licenses and access requirements
remain applicable. Review the repository README before working on model-backed
features.

For frontend work, use a Node.js version compatible with the declared Vite
dependency, then install dependencies in `segagent/frontend`:

```bash
npm install
```

## Validate the change

Choose checks appropriate to the files you changed. Documentation-only changes
do not require downloading weights, starting a server, or running models. Check
links, commands, and architecture descriptions against the source instead.

The test suite covers deterministic behavior including planner contracts,
dialogue, intent gates, prompt provenance, schemas, storage, retrieval, QC,
verification, and workflow control without downloading model weights. With
development dependencies installed, run from `segagent/`:

```bash
python -m pytest -q
```

For changes to the React console, run from `segagent/frontend/`:

```bash
npm run lint
npm run build
```

These commands describe contributor checks; they do not indicate that a
particular pull request has passed. Include the checks you actually ran, their
results, and any relevant checks you could not run in your pull request.

For behavior changes, add focused tests where practical. Model-backed changes
should describe the model revision, input characteristics, environment, and
observed result using synthetic or appropriately de-identified data. Keep
research claims proportionate to the evidence; unit tests do not establish
clinical validity or end-to-end model quality.

## Protect data and provenance

- Do not commit patient images, protected health information, identifying
  screenshots, credentials, downloaded model weights, or local case databases.
- Use synthetic examples in issues, pull requests, fixtures, and screenshots.
- Preserve image geometry, artifact provenance, source attribution, intent
  gates, and clarification boundaries when changing segmentation, dialogue, or QC.
- Cite the source and applicable terms when adding protocol knowledge or other
  third-party material. The repository's MIT license does not replace those terms.

Report security vulnerabilities privately using [SECURITY.md](SECURITY.md).

## Submit a pull request

Create a branch in your fork and submit a pull request against `main`.
Explain the problem, resulting behavior, and
validation. Include screenshots for UI changes only when they contain synthetic
or de-identified content. Update documentation when user-visible behavior or
configuration changes.

Contributions to SegAgent's own code are submitted under the repository's
[MIT license](LICENSE). Retain existing copyright and license notices, and do not
add material you lack permission to distribute.
