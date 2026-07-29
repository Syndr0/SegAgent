from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .knowledge import HybridKnowledgeBase, normalize
from .schemas import PlannerAction, PlannerDecision, ToolObservation
from .storage import ResearchStore


PLANNER_SYSTEM = """You are SegAgent, an evidence-first planner and assistant for one
3D medical-image case. You are in a continuing conversation about a single
patient. Select exactly one next action, and return a concise rationale_summary
naming the evidence needed or used.

Available actions:
- lookup_protocol: set site_query for OAR/treatment-site questions.
- segment: set structures to segment. Use real anatomical names; for a tumor
  target, use the full spatial description the user gave.
- inspect_mask: set structures to the names of contours that ALREADY exist in
  this case, to measure them. Prefer this over segment when the user refers to a
  contour that was already produced.
- ask_user: set user_message when you need information only the user has.
- final: set final_answer.

Rules:
- Never invent a measurement, protocol, citation, or completed tool call.
- Numbers in final_answer must come from tool observations. If you have not
  measured something, do not state a value for it.
- You may answer descriptive questions about the image directly in final_answer
  (what the scan shows, which structures are visible, where something lies)
  without running a tool, as long as you state no invented measurements.
- Do not diagnose disease, stage a tumor, or judge whether a plan is correct.
  You describe and measure what is present; a clinician interprets it.
- If evidence is insufficient, ask or explicitly state the limitation.
- Output one JSON object matching the supplied JSON schema and no extra text.
"""

# Fields the model must not set: they are filled by deterministic code, and the
# reasoning is captured from the model's thinking channel instead.
_MODEL_OWNED_EXCLUDED = ("gate", "thinking")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_thinking(raw: str) -> tuple[str, str]:
    """Split a thinking-model response into (reasoning, answer payload).

    Models with a thinking mode emit their reasoning inside <think>...</think>
    before the JSON. The reasoning is shown to the user as the agent's thought
    process; only the remainder is parsed as the decision.
    """
    text = raw or ""
    if THINK_CLOSE in text:
        head, _, tail = text.partition(THINK_CLOSE)
        return head.replace(THINK_OPEN, "").strip(), tail.strip()
    if text.lstrip().startswith(THINK_OPEN):
        # Thinking was truncated before it closed; there is no payload yet.
        return text.lstrip()[len(THINK_OPEN):].strip(), ""
    return "", text.strip()


class Planner(Protocol):
    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
        history: list[dict] | None = None,
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
        history: list[dict] | None = None,
    ) -> PlannerDecision:
        del case_id, step, history
        used = {item.tool for item in observations}
        lowered = normalize(question)
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
        enable_thinking: bool = True,
    ):
        self.model_id = model_id
        self.store = store
        self.montage_slices = montage_slices
        self.max_new_tokens = max_new_tokens
        self.device_name = device
        self.enable_thinking = enable_thinking
        self.model = None
        self.processor = None

    def _ensure(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor

        device = "cuda" if self.device_name == "auto" and torch.cuda.is_available() else self.device_name
        if device == "auto":
            device = "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        # Loaded through the Auto class so the planner model is purely a config
        # value: any vision-language checkpoint can be swapped in without a code
        # change, and bf16 is used on GPU.
        try:
            from transformers import AutoModelForImageTextToText as _AutoVL
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVL
        self.model = _AutoVL.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
            trust_remote_code=True,
        )

    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
        history: list[dict] | None = None,
    ) -> PlannerDecision:
        from .imaging import grounding_views, load_volume

        self._ensure()
        case = self.store.get_case(case_id)
        volume = load_volume(self.store.artifact_path(case.image))
        views = grounding_views(volume.data, self.montage_slices)
        schema = PlannerDecision.model_json_schema()
        for field in _MODEL_OWNED_EXCLUDED:
            schema.get("properties", {}).pop(field, None)
        state = {
            "step": step,
            "question": question,
            # Earlier turns for this patient, so references such as "the mask you
            # just made" resolve instead of being re-segmented.
            "conversation": [
                {"role": item.get("role"), "text": item.get("text")}
                for item in (history or [])[-12:]
            ],
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
            reasoning, payload = split_thinking(raw)
            try:
                decision = PlannerDecision.model_validate(self._decode_object(payload or raw))
                # The model must not set these; reasoning comes from its own
                # thinking channel, and gates are deterministic.
                decision.gate = None
                decision.thinking = reasoning or None
                return decision
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

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Chat templates without a thinking switch ignore the argument.
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
