from __future__ import annotations

import json
from threading import Thread
from typing import Any, Callable, Protocol

from .intent import match_canonical_terms
from .knowledge import HybridKnowledgeBase, normalize
from .schemas import PlannerAction, PlannerAnswer, PlannerDecision, ToolObservation
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
- run_qc: check uploaded contours against an independent model contour of the
  same structures; set site_query when you know the treatment site.
- ask_user: set user_message when you need information only the user has.
- final: set final_answer.

Workflows:
- Organs at risk without a stated treatment site: look at the image, propose the
  most likely site, and use ask_user to confirm it ("This looks like a thorax
  scan — shall I contour the thoracic OAR set?"). Put the proposed canonical
  site in site_query even though the action is ask_user. Do not contour before
  the user agrees. Once a site is confirmed, use lookup_protocol and segment
  its list.
- Tumor target: the prompt you pass to segment should be the user's spatial
  description, not the bare word "tumor" — the description is what makes the
  segmentation work. After the user confirms a first attempt is in the right
  region, segment the neighbouring organs, then re-segment the target with a
  prompt that states its position relative to those structures.
- A structure the user simply names: segment it directly.
- Contour check: run_qc, then report what it measured. Say that the comparison
  model is a second opinion, not ground truth: describe disagreement and let the
  clinician judge. Target volumes (GTV/CTV/PTV) are not compared — report their
  size and position, and list protocol structures that have no contour as a
  factual set difference, not as a verdict on the plan.
- Report: use inspect_mask (and run_qc when contours were uploaded) to gather
  real measurements first, then summarise sizes, positions and spatial
  relationships. Never state a measurement you have not obtained from a tool.

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
_MODEL_OWNED_EXCLUDED = ("gate", "thinking", "workflow_stage")

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


class ThinkingDeltaParser:
    """Extract only live text inside ``<think>`` from arbitrary token chunks."""

    def __init__(self, emit: Callable[[str], None] | None = None):
        self.emit = emit
        self.raw = ""
        self.emitted = 0
        self.closed = False

    def feed(self, chunk: str) -> None:
        if not chunk or self.closed:
            return
        self.raw += chunk
        start = self.raw.find(THINK_OPEN)
        if start < 0:
            return
        body_start = start + len(THINK_OPEN)
        close = self.raw.find(THINK_CLOSE, body_start)
        body_end = close if close >= 0 else len(self.raw)
        if close < 0:
            # Hold a possible partial closing tag until the next token chunk.
            for size in range(min(len(THINK_CLOSE) - 1, body_end - body_start), 0, -1):
                if self.raw.endswith(THINK_CLOSE[:size]):
                    body_end -= size
                    break
        body = self.raw[body_start:body_end]
        if len(body) > self.emitted:
            delta = body[self.emitted:]
            self.emitted = len(body)
            if self.emit:
                self.emit(delta)
        self.closed = close >= 0


VQA_SYSTEM = """You are SegAgent, looking at one 3D medical image with a clinician.
Answer their question about what the image shows, in a few plain sentences.

- Describe what is visible: modality, body region, and the major structures in
  view. Name the contours already on the case when they are relevant.
- State no measurement you were not given. You may say a structure is present or
  where it lies, but not how large it is unless a measurement is supplied.
- Do not diagnose, stage, or judge whether anything is abnormal. Describe.
- If you cannot tell from the views, say so.
Reply with prose only. No JSON, no headings.
"""


class Planner(Protocol):
    def decide(
        self,
        case_id: str,
        question: str,
        step: int,
        observations: list[ToolObservation],
        history: list[dict] | None = None,
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
    ) -> PlannerDecision: ...

    def answer(
        self,
        case_id: str,
        question: str,
        history: list[dict] | None = None,
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
    ) -> PlannerAnswer: ...


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
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
    ) -> PlannerDecision:
        del case_id, step, history, context, on_thinking_delta
        used = {item.tool for item in observations}
        lowered = normalize(question)
        asks_qc = (
            any(term in lowered for term in ("qc", "quality control", "quality-control", "质控"))
            or (
                any(term in lowered for term in ("check", "audit", "review", "verify", "inspect"))
                and any(
                    term in lowered
                    for term in ("contour", "contours", "segmentation", "structures")
                )
            )
        )
        if asks_qc and "run_qc" not in used:
            return PlannerDecision(
                action=PlannerAction.RUN_QC,
                rationale_summary="Run contour QC before summarizing the uploaded structures.",
                confidence=0.95,
            )
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

        explicit = match_canonical_terms(lowered, self.organ_names)
        if explicit and "segment" not in used:
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="The question explicitly names structures that can be measured.",
                confidence=0.95,
                structures=explicit,
            )
        if used.intersection(
            {
                "segment",
                "lookup_protocol",
                "inspect_mask",
                "run_qc",
                "build_report_evidence",
            }
        ):
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

    def answer(
        self,
        case_id: str,
        question: str,
        history: list[dict] | None = None,
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
    ) -> PlannerAnswer:
        del case_id, question, history, context, on_thinking_delta
        return PlannerAnswer(
            text=(
                "I cannot describe the image without a vision model configured. "
                "Name a structure and I can contour and measure it."
            )
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
        max_new_tokens: int = 2048,
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
            # Qwen3.5's current Transformers integration is registered here.
            from transformers import AutoModelForMultimodalLM as _AutoVL
        except ImportError:
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
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
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
            "turn_context": context or {},
            "artifact_inventory": self.store.artifact_inventory(case_id),
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
        self._append_artifact_views(content, case, volume, context)
        content.append(
            {"type": "text", "text": json.dumps(state, ensure_ascii=False, default=str)}
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": content},
        ]
        last_error = ""
        collected_reasoning: list[str] = []
        for attempt in range(2):
            parser = ThinkingDeltaParser(on_thinking_delta)
            raw = self._generate(
                messages,
                on_text_delta=parser.feed,
                # Preserve thinking on the first attempt. If it consumed the
                # whole token budget before JSON, retry schema repair without a
                # second long thought while retaining/displaying the first.
                enable_thinking=self.enable_thinking if attempt == 0 else False,
            )
            reasoning, payload = split_thinking(raw)
            if reasoning:
                collected_reasoning.append(reasoning)
            try:
                decision = PlannerDecision.model_validate(self._decode_object(payload or raw))
                # The model must not set these; reasoning comes from its own
                # thinking channel, and gates are deterministic.
                decision.gate = None
                decision.workflow_stage = None
                decision.thinking = "\n\n".join(collected_reasoning) or None
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

    def answer(
        self,
        case_id: str,
        question: str,
        history: list[dict] | None = None,
        context: dict | None = None,
        on_thinking_delta: Callable[[str], None] | None = None,
    ) -> PlannerAnswer:
        """Answer a descriptive question about the image, in prose.

        This is a separate path from decide(): a question about what the scan
        shows should produce a description, not an action in a tool schema.
        """
        from .imaging import grounding_views, load_volume

        self._ensure()
        case = self.store.get_case(case_id)
        volume = load_volume(self.store.artifact_path(case.image))
        contours = [
            item.label for item in case.artifacts if item.kind in {"mask", "contour"}
        ]
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Views of the case:"}
        ]
        for label, image in grounding_views(volume.data, self.montage_slices):
            content.extend([
                {"type": "text", "text": label},
                {"type": "image", "image": image},
            ])
        self._append_artifact_views(content, case, volume, context)
        if contours:
            content.append(
                {
                    "type": "text",
                    "text": "Contours already on this case: " + ", ".join(dict.fromkeys(contours)),
                }
            )
        for turn in (history or [])[-8:]:
            role = turn.get("role")
            text = (turn.get("text") or "").strip()
            if text:
                content.append({"type": "text", "text": f"{role}: {text}"})
        content.append({"type": "text", "text": f"Question: {question}"})

        messages = [
            {"role": "system", "content": VQA_SYSTEM},
            {"role": "user", "content": content},
        ]
        parser = ThinkingDeltaParser(on_thinking_delta)
        raw = self._generate(
            messages,
            on_text_delta=parser.feed,
            enable_thinking=self.enable_thinking,
        )
        reasoning, prose = split_thinking(raw)
        if not prose and reasoning:
            prose = self._generate(messages, enable_thinking=False)
        return PlannerAnswer(
            text=(prose or raw).strip(),
            thinking=reasoning or None,
        )

    def _append_artifact_views(self, content, case, volume, context: dict | None) -> None:
        """Attach only explicitly referenced/visible case artifacts to the VLM."""

        from .imaging import load_mask, mask_grounding_views

        ordered_ids = self._artifact_view_ids(case, context)
        by_id = {item.artifact_id: item for item in case.artifacts}
        for artifact_id in ordered_ids:
            artifact = by_id.get(artifact_id)
            if artifact is None:
                continue
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Referenced artifact id={artifact.artifact_id}, "
                        f"label={artifact.label}, kind={artifact.kind}"
                    ),
                }
            )
            try:
                if artifact.kind in {"mask", "contour"}:
                    mask = load_mask(self.store.artifact_path(artifact), volume)
                    for label, image in mask_grounding_views(
                        volume.data, mask, min(3, self.montage_slices)
                    ):
                        content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": f"{artifact.label} ({artifact.artifact_id}), {label}",
                                },
                                {"type": "image", "image": image},
                            ]
                        )
                elif artifact.kind == "overlay":
                    from PIL import Image

                    content.append(
                        {
                            "type": "image",
                            "image": Image.open(self.store.artifact_path(artifact)),
                        }
                    )
            except Exception:
                continue

    @staticmethod
    def _artifact_view_ids(case, context: dict | None, limit: int = 8) -> list[str]:
        """Prioritize the exact masks named in a turn over ambient visibility.

        A full OAR viewer can contain 15–20 visible masks. Named targets,
        selected IDs, and explicit references are the semantic context; ambient
        visible masks are only a fallback when none of those resolves.
        """

        planner_context = context or {}
        message_context = planner_context.get("message_context") or planner_context
        target_names = {
            " ".join(str(item).casefold().split())
            for item in ((planner_context.get("intent") or {}).get("targets") or [])
            if str(item).strip()
        }
        target_ids: list[str] = []
        if target_names:
            superseded = {
                str(artifact.metadata.get("derived_from"))
                for artifact in case.artifacts
                if getattr(artifact, "metadata", {}).get("derived_from")
            }
            matched_labels: set[str] = set()
            for artifact in reversed(case.artifacts):
                if artifact.kind not in {"mask", "contour"}:
                    continue
                label = " ".join(artifact.label.casefold().split())
                if (
                    label in target_names
                    and label not in matched_labels
                    and artifact.artifact_id not in superseded
                ):
                    target_ids.append(artifact.artifact_id)
                    matched_labels.add(label)

        explicit = [
            message_context.get("selected_mask_id"),
            *(message_context.get("referenced_artifact_ids") or []),
            *target_ids,
        ]
        resolved = [item for item in explicit if item]
        ambient = (
            []
            if resolved or target_names
            else list(message_context.get("visible_mask_ids") or [])
        )
        return list(dict.fromkeys([*resolved, *ambient]))[:limit]

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

    def _generate(
        self,
        messages: list[dict],
        on_text_delta: Callable[[str], None] | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import TextIteratorStreamer

        thinking = self.enable_thinking if enable_thinking is None else enable_thinking

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
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
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        streamer=streamer,
                    )
            except BaseException as exc:  # surfaced on the caller thread below
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=generate, name="segagent-qwen-generate", daemon=True)
        worker.start()
        chunks: list[str] = []
        for chunk in streamer:
            chunks.append(chunk)
            if on_text_delta:
                on_text_delta(chunk)
        worker.join()
        if errors:
            raise RuntimeError(f"model generation failed: {errors[0]}") from errors[0]
        return "".join(chunks).strip()
