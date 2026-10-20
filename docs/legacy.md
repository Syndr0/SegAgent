# Original plugin integration

[Project overview](../README.md) · [Research v2 setup](getting-started.md)

The `segagent/` directory is the original SegAgent extension for [voxtell-web-plugin](https://github.com/gomesgustavoo/voxtell-web-plugin). It is retained for reference and existing integrations. New users should start with the standalone `segagent-research-v2/` application.

## What the extension contains

The extension combines a Qwen2.5-VL ReAct loop with the plugin's VoxTell predictor, chat and QC components, local knowledge, and contour-checking modules. Its server also contains DICOM conversion and RTSTRUCT-related routes that depend on functionality supplied by the base plugin.

It does **not** contain the complete base application. For example, `backend/dicom_sessions.py`, the frontend viewer, package metadata, and run scripts must come from a compatible plugin checkout. The Research v2 app does not use this overlay.

## Integration procedure

Use a separate working copy of the base plugin. Review the changes before replacing files, and record the upstream commit used for your experiment; compatibility with arbitrary future plugin versions is not guaranteed.

```bash
git clone https://github.com/gomesgustavoo/voxtell-web-plugin.git
git clone https://github.com/Syndr0/SegAgent.git
```

Complete the base plugin's own environment, dependency, and model-weight setup first. Then copy the **complete extension**, from the parent directory containing both checkouts:

```bash
cp -R SegAgent/segagent/backend/. voxtell-web-plugin/backend/
cp -R SegAgent/segagent/frontend/src/. voxtell-web-plugin/frontend/src/
```

Copying only `agent.py`, `server.py`, `ChatPanel.tsx`, and `App.tsx` is insufficient: the current extension also imports QC and knowledge modules. The directory overlay includes those modules while retaining base-plugin files not supplied here.

With the base plugin's Python environment active:

```bash
python -m pip install -r SegAgent/segagent/requirements-segagent.txt
cd voxtell-web-plugin
git diff --stat
```

Review the diff and use the startup instructions from that checkout. Do not substitute Research v2 commands such as `uvicorn backend.api:app`; this extension's entry points and module imports are different. Verify the base frontend provides the dependencies imported by the copied components.

## Interface differences

| Topic | Original extension | Research v2 |
| --- | --- | --- |
| Planner | Free-form ReAct output | Validated `PlannerDecision` |
| Requests | `/chat`, `/predict`, `/qc` | Case/run API under `/api/` |
| Mask access | `/mask/{mask_id}` | Case-scoped artifact endpoints |
| DICOM/RTSTRUCT | Base-plugin-dependent integration | Not accepted by the v2 upload API |
| Durable graph review | Not the v2 workflow | SQLite checkpoint and segmentation review |

Environment names also differ. The legacy planner reads `SEGAGENT_LLM`, while v2 uses `SEGAGENT_LLM_MODEL`. Read the legacy modules for their settings rather than applying the v2 configuration table to them.

## Scope

This documentation preserves an integration path; it is not a compatibility or clinical-validation claim. Use de-identified research data and review all masks and QC findings. Model weights and upstream code retain their own licenses.
