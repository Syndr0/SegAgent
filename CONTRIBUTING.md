# Contributing to SegAgent

SegAgent is research software for agentic 3D medical-image segmentation and
contour quality control. Contributions to the code, documentation, evaluation
methods, and reproducible research workflows are welcome.

## Discuss the change

Search existing [issues](https://github.com/Syndr0/SegAgent/issues) before opening
a new one. For a substantial feature, architecture change, or new dependency,
open an issue to discuss the intended behavior and scope before implementation.
Small documentation corrections can go directly to a pull request.

State which implementation your change targets:

- `segagent-research-v2/`: the research implementation, including the typed
  backend, durable workflow, research console, tests, and evaluation harness.
- `segagent/`: the legacy implementation, retained for reference and comparison.

Keep changes focused on one implementation unless a shared change is necessary.
Do not imply that a legacy feature is available in v2 without checking its code.

## Set up a development environment

Use Python 3.11 or newer for v2. From the repository root:

```bash
cd segagent-research-v2
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
dependency, then install dependencies in `segagent-research-v2/frontend`:

```bash
npm install
```

## Validate the change

Choose checks appropriate to the files you changed. Documentation-only changes
do not require downloading weights, starting a server, or running models. Check
links, commands, and architecture descriptions against the source instead.

The v2 test suite covers deterministic planner, schema, storage, and knowledge
behavior without requiring model weights. With development dependencies
installed, run from `segagent-research-v2/`:

```bash
python -m pytest -q
```

For changes to the React console, run from `segagent-research-v2/frontend/`:

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
- Preserve image geometry, artifact provenance, source attribution, and human
  review boundaries when changing segmentation or QC behavior.
- Cite the source and applicable terms when adding protocol knowledge or other
  third-party material. The repository's MIT license does not replace those terms.

Report security vulnerabilities privately using [SECURITY.md](SECURITY.md).

## Submit a pull request

Create a branch in your fork and submit a pull request against the relevant
branch of this repository. Explain the problem, resulting behavior, and
validation. Include screenshots for UI changes only when they contain synthetic
or de-identified content. Update documentation when user-visible behavior or
configuration changes.

Contributions to SegAgent's own code are submitted under the repository's
[MIT license](LICENSE). Retain existing copyright and license notices, and do not
add material you lack permission to distribute.
