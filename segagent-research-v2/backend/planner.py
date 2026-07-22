from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .knowledge import HybridKnowledgeBase, normalize
from .schemas import PlannerAction, PlannerDecision, ToolObservation
from .storage import ResearchStore


PLANNER_SYSTEM = """You are SegAgent Research v2, an evidence-first planner for one
3D medical-image case. Select exactly one next action. Do not provide hidden
chain-of-thought. Return a concise rationale_summary that names the evidence
needed or used.

Available actions:
- lookup_protocol: set site_query for OAR/treatment-site questions.
- segment: set structures to real anatomical names.
- run_qc: use only when uploaded contours should be audited.
- ask_user: set user_message when the case/question is genuinely ambiguous.
- final: set final_answer using only the supplied tool observations.

Rules:
- Never invent a measurement, protocol, citation, or completed tool call.
- Do not diagnose disease from a segmentation volume alone.
- If evidence is insufficient, ask or explicitly state the limitation.
- Output one JSON object matching the supplied JSON schema and no extra text.
"""


class Planner(Protocol):
    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
    ) -> PlannerDecision: ...


class RulePlanner:
    """Deterministic baseline for ablation tests and no-model development."""

    def __init__(self, knowledge: HybridKnowledgeBase):
        self.knowledge = knowledge
        self.organ_names = sorted(
            {
                organ.casefold()
                for protocol in knowledge.protocols
                for organ in protocol.get("oars", [])
            },
            key=len,
            reverse=True,
        )

    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
    ) -> PlannerDecision:
        del case_id, step
        used = {item.tool for item in observations}
        lowered = normalize(question)
        rejected = next(
            (
                item
                for item in observations
                if item.tool == "evidence_critic"
                and item.data.get("approval", {}).get("decision") in {"reject", "feedback"}
            ),
            None,
        )
        if rejected is not None:
            return PlannerDecision(
                action=PlannerAction.ASK_USER,
                rationale_summary="Human review did not approve the generated mask evidence.",
                confidence=0.99,
                user_message=(
                    "The generated masks were not approved. Please revise the requested "
                    "structures or provide reviewer guidance before another segmentation run."
                ),
            )
        if any(term in lowered for term in ("quality control", "qc", "audit contour", "check contour")):
            if "run_qc" not in used:
                return PlannerDecision(
                    action=PlannerAction.RUN_QC,
                    rationale_summary="Uploaded contours require deterministic and expert-model review.",
                    confidence=0.98,
                )
            return self._final_from_observations(observations)

        asks_protocol = any(
            term in lowered
            for term in ("oar", "organs at risk", "radiotherapy", "radiation plan", "contour all")
        )
        if asks_protocol and "lookup_protocol" not in used:
            return PlannerDecision(
                action=PlannerAction.LOOKUP_PROTOCOL,
                rationale_summary="A curated site protocol is required before selecting OAR structures.",
                confidence=0.92,
                site_query=question,
            )
        if asks_protocol and "segment" not in used:
            protocol_observation = next(
                (item for item in observations if item.tool == "lookup_protocol"), None
            )
            if protocol_observation:
                structures = list(protocol_observation.data.get("oars", []))
                if structures:
                    return PlannerDecision(
                        action=PlannerAction.SEGMENT,
                        rationale_summary="Segment the OAR list returned by the selected protocol.",
                        confidence=0.9,
                        structures=structures,
                    )

        explicit = [name for name in self.organ_names if name in lowered]
        if explicit and "segment" not in used:
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="The question explicitly names structures that can be measured.",
                confidence=0.95,
                structures=explicit,
            )
        if "segment" in used or "lookup_protocol" in used:
            return self._final_from_observations(observations)
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="No supported structure or treatment site was identified reliably.",
            confidence=0.35,
            user_message=(
                "Please name the anatomical structure or radiotherapy site you want "
                "to analyze (for example, left kidney, thorax OARs, or pelvic OARs)."
            ),
        )

    @staticmethod
    def _final_from_observations(observations: list[ToolObservation]) -> PlannerDecision:
        lines = [item.summary for item in observations]
        return PlannerDecision(
            action=PlannerAction.FINAL,
            rationale_summary="The available tool observations are sufficient for an evidence summary.",
            confidence=0.88,
            final_answer="\n\n".join(lines),
        )


class QwenStructuredPlanner:
    """Local multimodal planner with Pydantic-validated JSON decisions.

    The adapter uses schema-guided prompting and retries once on invalid JSON.
    A local vLLM structured-output endpoint can replace this class without
    changing the workflow or tools.
    """

    def __init__(
        self,
        model_id: str,
        store: ResearchStore,
        montage_slices: int = 6,
        max_new_tokens: int = 512,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.store = store
        self.montage_slices = montage_slices
        self.max_new_tokens = max_new_tokens
        self.device_name = device
        self.model = None
        self.processor = None

    def _ensure(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = "cuda" if self.device_name == "auto" and torch.cuda.is_available() else self.device_name
        if device == "auto":
            device = "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
        )

    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
    ) -> PlannerDecision:
        from .imaging import grounding_views, load_volume

        self._ensure()
        case = self.store.get_case(case_id)
        volume = load_volume(self.store.artifact_path(case.image))
        views = grounding_views(volume.data, self.montage_slices)
        schema = PlannerDecision.model_json_schema()
        state = {
            "step": step,
            "question": question,
            "observations": [item.model_dump(mode="json") for item in observations],
            "required_schema": schema,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Grounding views from one case, followed by current typed state.",
            }
        ]
        for label, image in views:
            content.extend(
                [
                    {"type": "text", "text": label},
                    {"type": "image", "image": image},
                ]
            )
        for artifact in case.artifacts[-8:]:
            if artifact.kind == "overlay":
                try:
                    from PIL import Image

                    content.extend(
                        [
                            {"type": "text", "text": f"Prior mask evidence: {artifact.label}"},
                            {"type": "image", "image": Image.open(self.store.artifact_path(artifact))},
                        ]
                    )
                except Exception:
                    continue
        content.append(
            {"type": "text", "text": json.dumps(state, ensure_ascii=False, default=str)}
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": content},
        ]
        last_error = ""
        for _ in range(2):
            raw = self._generate(messages)
            try:
                return PlannerDecision.model_validate(self._decode_object(raw))
            except Exception as exc:
                last_error = str(exc)
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your output did not validate. Return only one corrected JSON object. "
                            f"Validation error: {last_error}"
                        ),
                    }
                )
        raise RuntimeError(f"planner failed structured-output validation: {last_error}")

    @staticmethod
    def _decode_object(text: str) -> dict:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("no JSON object in planner response")

    def _generate(self, messages: list[dict]) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
