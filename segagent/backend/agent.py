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
# How many mask-overlay slices to feed back to the LLM after each segmentation
# (the `I_k` image observation in the design). Kept small to bound context growth.
N_OVERLAY_SLICES = int(os.environ.get("SEGAGENT_OVERLAY_SLICES", "3"))
MAX_NEW_TOKENS = int(os.environ.get("SEGAGENT_MAX_NEW_TOKENS", "512"))

SYSTEM_PROMPT = """You are SegAgent, a careful radiology reasoning assistant that \
analyzes a single 3D medical scan (CT / MRI / PET) to answer the user's question.

You cannot see the full 3D volume directly. You are shown a few representative 2D \
slices for grounding, and you have TWO tools:

  lookup_oar(query): look up the standard organs-at-risk (OAR) list for a \
radiotherapy site or clinical description in a curated knowledge base. Use it \
whenever the user asks to contour/segment "OARs" / "risk organs", or names a \
treatment site or diagnosis (e.g. "head and neck", "prostate plan", "鼻咽癌") \
without spelling out every organ. It returns the exact organ names to delineate.

  segment(prompt): runs an expert 3D segmentation model (VoxTell) on the whole \
volume. `prompt` is one or more anatomical structures; to segment several at \
once, separate them with semicolons, e.g. segment("liver; spleen; left kidney"). \
It returns BOTH (a) quantitative statistics per structure (voxel count, volume in \
mm^3, whether found, mean intensity) AND (b) images showing the produced mask(s) \
as a red overlay on the slices where they are largest. Look at the overlay to \
judge whether the segmentation is correct and to reason about location and shape.

Use anatomical terms the model understands, e.g. "liver", "left kidney", "spleen", \
"right lung upper lobe", "L4 vertebra", "prostate tumor".

Work step by step. On EACH turn reply in EXACTLY one of these two forms:

  THOUGHT: <your reasoning for this step>
  ACTION: <tool>(...)          where <tool> is lookup_oar or segment

or, once you have enough evidence:

  THOUGHT: <your reasoning>
  FINAL: <your complete answer to the user>

Typical OAR workflow: first lookup_oar to get the organ list, then ONE segment \
call with all of those organs (semicolon-separated), then a FINAL summary.

Rules:
- Exactly one ACTION per turn. Do not invent statistics or organ lists — only use \
values returned in OBSERVATION messages.
- If a structure returns 0 voxels, it was not found; reason about that.
- Keep going until you can justify the answer, then give FINAL.
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
                 device: Optional[torch.device] = None, knowledge_base=None):
        self.predictor = predictor
        self.write_seg = write_seg
        self.mask_dir = mask_dir
        # Optional OARKnowledgeBase for the lookup_oar tool (site -> organ list).
        self.kb = knowledge_base
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

    # -------------------------------------------------------------- imaging
    @staticmethod
    def _as_3d(volume: np.ndarray) -> np.ndarray:
        vol = np.asarray(volume, dtype=np.float32)
        return vol[0] if vol.ndim == 4 else vol  # (C,X,Y,Z) -> (X,Y,Z)

    @staticmethod
    def _window_bounds(vol: np.ndarray):
        """Modality-agnostic intensity window (1st/99th percentile)."""
        lo, hi = np.percentile(vol, [1.0, 99.0])
        if hi <= lo:
            hi = float(lo) + 1.0
        return float(lo), float(hi)

    @staticmethod
    def _slice_to_uint8(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
        sl = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
        return (sl * 255).astype(np.uint8)

    def _to_pil_slices(self, volume: np.ndarray, n: int):
        """Turn a 3D volume into `n` evenly spaced, windowed axial PIL images."""
        from PIL import Image

        vol = self._as_3d(volume)
        lo, hi = self._window_bounds(vol)
        depth = vol.shape[-1]
        idxs = np.linspace(depth * 0.15, depth * 0.85, n).round().astype(int)
        idxs = np.clip(idxs, 0, depth - 1)
        images = []
        for z in idxs:
            sl = self._slice_to_uint8(vol[..., z], lo, hi)
            # Orient so rows read top-to-bottom nicely; harmless for grounding.
            images.append(Image.fromarray(sl.T[::-1]).convert("RGB"))
        return images

    def _overlay_slices(self, volume: np.ndarray, mask: np.ndarray, n: int):
        """Render the mask (red, 50%) over the `n` slices with the most mask.

        This is the `I_k` image observation from the design: the actual
        segmentation matrix, handed back to the LLM visually so it can reason
        about the structure's location, shape and extent — not just its stats.
        """
        from PIL import Image

        vol = self._as_3d(volume)
        m = np.asarray(mask) > 0
        lo, hi = self._window_bounds(vol)

        # Choose the axial slices where the mask has the largest cross-section.
        areas = m.sum(axis=(0, 1))  # per-z voxel counts
        zs = [int(z) for z in np.argsort(areas)[::-1] if areas[z] > 0][:n]
        zs.sort()

        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        images = []
        for z in zs:
            base = self._slice_to_uint8(vol[..., z], lo, hi).astype(np.float32)
            rgb = np.stack([base, base, base], axis=-1)  # (X,Y,3) grayscale
            msl = m[..., z]
            rgb[msl] = 0.5 * rgb[msl] + 0.5 * red        # blend mask in red
            rgb = rgb.astype(np.uint8)
            # Same orientation transform as _to_pil_slices, applied to (X,Y,3).
            rgb = np.transpose(rgb, (1, 0, 2))[::-1]
            images.append(Image.fromarray(rgb))
        return images

    # ---------------------------------------------------------------- tool
    def _segment(self, image_np: np.ndarray, props, prompt: str):
        """Run VoxTell for one or more (';'-separated) structures in ONE pass.

        Returns ``(observation_text, masks, overlay_images)`` where ``masks`` is
        a list of ``{"mask_id", "prompt"}`` for each structure that was found,
        and ``overlay_images`` are the produced masks rendered on the image for
        the LLM to reason over.
        """
        structures = [s.strip() for s in re.split(r"[;\n]+", prompt) if s.strip()]
        if not structures:
            structures = [prompt.strip()]

        # A single VoxTell inference handles all prompts at once (num_prompts,X,Y,Z).
        seg = self.predictor.predict_single_image(image_np, structures)

        img3d = self._as_3d(image_np)
        spacing = None
        try:
            spacing = props.get("spacing") if isinstance(props, dict) else None
        except Exception:
            spacing = None
        vox_mm3 = (float(abs(np.prod(list(spacing)[:3])))
                   if spacing is not None and len(spacing) >= 3 else None)

        # Overlay budget: full detail for a single structure, one slice each when
        # batching, capped so a large OAR set does not blow up the LLM context.
        per_struct = N_OVERLAY_SLICES if len(structures) == 1 else 1
        overlay_cap = max(N_OVERLAY_SLICES, 6)

        lines, masks, overlays = [], [], []
        for i, name in enumerate(structures):
            mask = np.asarray(seg[i]).astype(np.uint8)
            voxels = int(mask.sum())
            if voxels == 0:
                lines.append(f'- "{name}": NOT FOUND (0 voxels).')
                continue

            if vox_mm3 is not None:
                vol_str = f"{voxels} vox (~{voxels * vox_mm3 / 1000.0:.1f} mL)"
            else:
                vol_str = f"{voxels} vox"
            mean_int = float(img3d[mask > 0].mean())
            lines.append(f'- "{name}": volume={vol_str}, mean_intensity={mean_int:.1f}.')

            mask_id = uuid.uuid4().hex[:12]
            out_path = os.path.join(self.mask_dir, f"{mask_id}.nii.gz")
            try:
                self.write_seg(mask, out_path, props)
                masks.append({"mask_id": mask_id, "prompt": name})
            except Exception:
                pass  # stats still useful even if writing failed

            if len(overlays) < overlay_cap:
                try:
                    overlays += self._overlay_slices(image_np, mask, per_struct)
                except Exception:
                    pass

        header = (f'segment("{prompt}") -> {len(masks)}/{len(structures)} '
                  f"structure(s) found.")
        obs = header + "\n" + "\n".join(lines)
        return obs, masks, overlays[:overlay_cap]

    # --------------------------------------------------------------- parse
    @staticmethod
    def _parse(text: str):
        """Return ('final', answer) | ('action', (tool, arg)) | ('final', text)."""
        final = re.search(r"FINAL:\s*(.+)", text, re.S | re.I)
        action = re.search(r"ACTION:\s*(segment|lookup_oar)\s*\(\s*(.+?)\s*\)",
                           text, re.S | re.I)
        # A FINAL that appears before any ACTION wins.
        if final and (not action or final.start() < action.start()):
            return "final", final.group(1).strip()
        if action:
            tool = action.group(1).lower()
            arg = action.group(2).strip().strip('"').strip("'").strip()
            return "action", (tool, arg)
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

            tool, arg = payload

            # Knowledge-base lookup: return the curated OAR list as an observation.
            if tool == "lookup_oar":
                yield {"type": "action", "step": step, "tool": "lookup_oar",
                       "prompt": arg}
                if self.kb is not None:
                    obs = self.kb.format_observation(arg)
                else:
                    obs = (f'lookup_oar("{arg}") -> knowledge base unavailable. '
                           f"Call segment() with explicit organ names.")
                yield {"type": "observation", "step": step, "text": obs,
                       "prompt": arg}
                messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})
                continue

            # Segmentation expert (single structure or ';'-separated batch).
            yield {"type": "action", "step": step, "tool": "segment", "prompt": arg}
            try:
                obs, masks, overlays = self._segment(image_np, props, arg)
            except Exception as e:
                obs, masks, overlays = f'segment("{arg}") -> ERROR: {e}', [], []

            yield {"type": "observation", "step": step, "text": obs, "prompt": arg}
            for m in masks:
                yield {"type": "mask", "step": step, "mask_id": m["mask_id"],
                       "prompt": m["prompt"]}

            # Feed stats + the rendered mask(s) back to the LLM (design's `I_k`).
            obs_content = [{"type": "text", "text": f"OBSERVATION: {obs}"}]
            if overlays:
                obs_content.append({
                    "type": "text",
                    "text": (f"Below are {len(overlays)} slice(s) with the "
                             f"segmentation shown as a red overlay. Check the "
                             f"location and shape."),
                })
                obs_content += [{"type": "image", "image": im} for im in overlays]
            messages.append({"role": "user", "content": obs_content})

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
