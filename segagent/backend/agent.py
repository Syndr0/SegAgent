"""SegAgent: a reasoning-segmentation agent built from pretrained models.

This realizes the bottom half of the architecture:

    Question ─┐
    Image ────┤→ [ Qwen2.5-VL-7B ]  reason → emit query Qk
                        │  ▲
                        ▼  │ observation Ik (stats)
                 [ VoxTell expert ] → 3D mask

Qwen2.5-VL is the planner ("Large Language Model" box). At every step it
either calls the `segment` tool (a free-text query handed to VoxTell, the
"Segmentation Expert Model") or produces the final answer. Both models are
pretrained and used as-is — Qwen runs zero-shot with a ReAct-style protocol,
VoxTell does the actual volumetric segmentation.

The run() method is a generator yielding events so the web layer can stream
the chain of thought live.
"""

import os
import re
import uuid
from typing import Callable, Iterator, List, Optional

import numpy as np
import torch

# Lazy heavy imports (transformers / qwen_vl_utils / PIL) happen inside
# _ensure_llm so the FastAPI process still boots even if the LLM weights are
# not present yet.

QWEN_MODEL_ID = os.environ.get("SEGAGENT_LLM", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_STEPS = int(os.environ.get("SEGAGENT_MAX_STEPS", "6"))
N_MONTAGE_SLICES = int(os.environ.get("SEGAGENT_MONTAGE_SLICES", "6"))
MAX_NEW_TOKENS = int(os.environ.get("SEGAGENT_MAX_NEW_TOKENS", "512"))

SYSTEM_PROMPT = """You are SegAgent, a careful radiology reasoning assistant that \
analyzes a single 3D medical scan (CT / MRI / PET) to answer the user's question.

You cannot see the full 3D volume directly. You are shown a few representative 2D \
slices for grounding, and you have ONE tool:

  segment(prompt): runs an expert 3D segmentation model (VoxTell) on the whole \
volume. `prompt` is a short free-text anatomical description. It returns \
quantitative statistics about the segmented region (voxel count, volume in mm^3, \
whether anything was found, bounding box, mean image intensity inside the mask).

Use anatomical terms the model understands, e.g. "liver", "left kidney", "spleen", \
"right lung upper lobe", "L4 vertebra", "prostate tumor".

Work step by step. On EACH turn reply in EXACTLY one of these two forms:

  THOUGHT: <your reasoning for this step>
  ACTION: segment("<free-text query>")

or, once you have enough evidence:

  THOUGHT: <your reasoning>
  FINAL: <your complete answer to the user>

Rules:
- Exactly one ACTION per turn. Do not invent statistics — only use values returned \
in OBSERVATION messages.
- If a structure returns 0 voxels, it was not found; reason about that.
- Keep segmenting until you can justify the answer, then give FINAL.
"""


class SegAgent:
    """Drives Qwen2.5-VL over VoxTell in a ReAct loop.

    Parameters
    ----------
    predictor:
        A loaded ``VoxTellPredictor`` (shares the already-loaded expert model).
    write_seg:
        Callable ``(mask_np, out_path, props) -> None`` used to persist a mask
        as ``.nii.gz`` (pass ``NibabelIOWithReorient().write_seg``).
    mask_dir:
        Directory where produced masks are written so the web layer can serve
        them back to the viewer.
    device:
        Torch device for the LLM.
    """

    def __init__(self, predictor, write_seg: Callable, mask_dir: str,
                 device: Optional[torch.device] = None):
        self.predictor = predictor
        self.write_seg = write_seg
        self.mask_dir = mask_dir
        os.makedirs(mask_dir, exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    # ------------------------------------------------------------------ LLM
    def _ensure_llm(self):
        """Load Qwen2.5-VL on first use (kept out of server startup path)."""
        if self.model is not None:
            return
        from transformers import (AutoProcessor,
                                   Qwen2_5_VLForConditionalGeneration)

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        ).eval()
        # min/max pixels keep slice tokens bounded so context does not explode.
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)

    # -------------------------------------------------------------- montage
    @staticmethod
    def _to_pil_slices(volume: np.ndarray, n: int):
        """Turn a 3D volume into `n` evenly spaced, windowed axial PIL images."""
        from PIL import Image

        vol = np.asarray(volume, dtype=np.float32)
        if vol.ndim == 4:  # (C, X, Y, Z) -> drop channel
            vol = vol[0]
        # Modality-agnostic windowing on the whole volume.
        lo, hi = np.percentile(vol, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        depth = vol.shape[-1]
        idxs = np.linspace(depth * 0.15, depth * 0.85, n).round().astype(int)
        idxs = np.clip(idxs, 0, depth - 1)
        images = []
        for z in idxs:
            sl = vol[..., z]
            sl = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
            sl = (sl * 255).astype(np.uint8)
            # Orient so rows read top-to-bottom nicely; harmless for grounding.
            images.append(Image.fromarray(sl.T[::-1]).convert("RGB"))
        return images

    # ---------------------------------------------------------------- tool
    def _segment(self, image_np: np.ndarray, props, prompt: str):
        """Run VoxTell for one prompt; return (observation_text, mask_id or None)."""
        seg = self.predictor.predict_single_image(image_np, [prompt])  # (1,X,Y,Z)
        mask = np.asarray(seg[0]).astype(np.uint8)
        voxels = int(mask.sum())

        if voxels == 0:
            return (f'segment("{prompt}") -> NOT FOUND: 0 voxels segmented. '
                    f"The model did not localize this structure."), None

        # Physical volume if spacing is available.
        spacing = None
        try:
            spacing = props.get("spacing") if isinstance(props, dict) else None
        except Exception:
            spacing = None
        if spacing is not None and len(spacing) >= 3:
            vox_mm3 = float(abs(np.prod(list(spacing)[:3])))
            vol_ml = voxels * vox_mm3 / 1000.0
            vol_str = f"{voxels} voxels (~{vol_ml:.1f} mL)"
        else:
            vol_str = f"{voxels} voxels"

        coords = np.argwhere(mask > 0)
        bb_min = coords.min(0)
        bb_max = coords.max(0)
        bbox = f"[{bb_min.tolist()} .. {bb_max.tolist()}]"

        img3d = np.asarray(image_np)
        if img3d.ndim == 4:
            img3d = img3d[0]
        mean_int = float(img3d[mask > 0].mean())

        # Persist mask so the viewer can overlay it.
        mask_id = uuid.uuid4().hex[:12]
        out_path = os.path.join(self.mask_dir, f"{mask_id}.nii.gz")
        try:
            self.write_seg(mask, out_path, props)
        except Exception:
            mask_id = None  # observation still useful even if writing failed

        obs = (f'segment("{prompt}") -> FOUND. volume={vol_str}; '
               f"mean_intensity={mean_int:.1f}; bbox(voxels)={bbox}.")
        return obs, mask_id

    # --------------------------------------------------------------- parse
    @staticmethod
    def _parse(text: str):
        """Return ('final', answer) | ('action', query) | ('final', text)."""
        final = re.search(r"FINAL:\s*(.+)", text, re.S | re.I)
        action = re.search(r"ACTION:\s*segment\s*\(\s*(.+?)\s*\)",
                           text, re.S | re.I)
        # A FINAL that appears before any ACTION wins.
        if final and (not action or final.start() < action.start()):
            return "final", final.group(1).strip()
        if action:
            q = action.group(1).strip().strip('"').strip("'").strip()
            return "action", q
        # No protocol match: treat the whole thing as the answer.
        return "final", text.strip()

    @staticmethod
    def _thought(text: str) -> str:
        m = re.search(r"THOUGHT:\s*(.+?)(?:\n\s*(?:ACTION|FINAL):|$)",
                      text, re.S | re.I)
        return (m.group(1).strip() if m else text.strip())

    def _generate(self, messages) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()

    # ----------------------------------------------------------------- run
    def run(self, image_np: np.ndarray, props, question: str) -> Iterator[dict]:
        """Yield events: {type: thinking|action|observation|mask|answer|error}."""
        try:
            self._ensure_llm()
            slices = self._to_pil_slices(image_np, N_MONTAGE_SLICES)
        except Exception as e:  # model/deps not ready
            yield {"type": "error", "text": f"Failed to initialize agent: {e}"}
            return

        user_content = [{"type": "image", "image": im} for im in slices]
        user_content.append({
            "type": "text",
            "text": (f"These are {len(slices)} representative slices of one 3D "
                     f"scan.\n\nQuestion: {question}"),
        })
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        for step in range(1, MAX_STEPS + 1):
            try:
                reply = self._generate(messages)
            except Exception as e:
                yield {"type": "error", "text": f"LLM generation failed: {e}"}
                return

            messages.append({"role": "assistant", "content": reply})
            kind, payload = self._parse(reply)
            yield {"type": "thinking", "step": step, "text": self._thought(reply)}

            if kind == "final":
                yield {"type": "answer", "text": payload}
                return

            # kind == "action": call the segmentation expert.
            yield {"type": "action", "step": step, "prompt": payload}
            try:
                obs, mask_id = self._segment(image_np, props, payload)
            except Exception as e:
                obs, mask_id = f'segment("{payload}") -> ERROR: {e}', None

            yield {"type": "observation", "step": step, "text": obs,
                   "prompt": payload}
            if mask_id:
                yield {"type": "mask", "step": step, "mask_id": mask_id,
                       "prompt": payload}

            messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})

        # Ran out of steps — force a wrap-up answer.
        messages.append({"role": "user", "content":
                         "You have reached the step limit. Give your FINAL answer "
                         "now based on the observations so far."})
        try:
            reply = self._generate(messages)
            _, payload = self._parse(reply)
            yield {"type": "thinking", "step": MAX_STEPS + 1,
                   "text": self._thought(reply)}
            yield {"type": "answer", "text": payload}
        except Exception as e:
            yield {"type": "error", "text": f"LLM generation failed: {e}"}
