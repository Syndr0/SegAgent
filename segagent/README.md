# Original SegAgent plugin extension

This directory preserves the original SegAgent extension for an external [voxtell-web-plugin](https://github.com/gomesgustavoo/voxtell-web-plugin) checkout. It contains replacement backend modules, knowledge files, and frontend components, rather than a standalone application.

Use [Research v2](../segagent-research-v2/) for the independent FastAPI/React application. For this extension, follow the [legacy integration guide](../docs/legacy.md); it includes the complete overlay and explains the external files required by the server and viewer.

| Path | Contents |
| --- | --- |
| `backend/agent.py` | Qwen2.5-VL ReAct planner |
| `backend/server.py` | Web-plugin server extension |
| `backend/qc_*.py` | Contour ingestion, checks, metrics, and expert comparison |
| `backend/knowledge_base.py`, `backend/guideline_kb.py` | Organ/protocol registry and guideline lookup |
| `backend/knowledge/` | Prototype organ and OAR reference data |
| `frontend/src/` | Application, chat, and QC components |
| `requirements-segagent.txt` | Additional Python dependencies |

The extension and Research v2 use different APIs and settings. Features in the external plugin, including DICOM session handling and viewer dependencies, are not supplied by this directory alone.
