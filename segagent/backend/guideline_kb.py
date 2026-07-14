"""Optional guideline-document RAG for QC.

Chunks the `.md`/`.txt` files in ``knowledge/guidelines/`` and retrieves the most
relevant passages for a query, using the injected embedder (VoxTell's already-
loaded Qwen3-Embedding-4B). If the folder is empty the KB simply has no chunks
and the caller disables it. numpy-only; no torch import here.
"""

import glob
import os
import re
from typing import Callable, List, Optional

import numpy as np

RETRIEVE_MIN_SIM = 0.20


class GuidelineKB:
    def __init__(self, folder: str,
                 embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
                 chunk_chars: int = 600):
        self.embed_fn = embed_fn
        self.chunks: List[dict] = []   # {"text", "source"}
        self._embeds = None            # lazily computed (n_chunks, dim)
        if folder and os.path.isdir(folder):
            for path in sorted(glob.glob(os.path.join(folder, "**", "*"),
                                         recursive=True)):
                if os.path.isfile(path) and path.lower().endswith((".md", ".txt")):
                    self._add(path, chunk_chars)

    def _add(self, path: str, chunk_chars: int):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            return
        src = os.path.basename(path)
        for ch in self._chunk(text, chunk_chars):
            self.chunks.append({"text": ch, "source": src})

    @staticmethod
    def _chunk(text: str, chunk_chars: int) -> List[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks, cur = [], ""
        for p in paras:
            if len(cur) + len(p) + 1 <= chunk_chars:
                cur = (cur + "\n" + p).strip()
            else:
                if cur:
                    chunks.append(cur)
                    cur = ""
                if len(p) <= chunk_chars:
                    cur = p
                else:  # split an over-long paragraph
                    for i in range(0, len(p), chunk_chars):
                        chunks.append(p[i:i + chunk_chars])
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-8, None)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Top-k guideline passages for the query (empty if none/weak)."""
        if not self.chunks or self.embed_fn is None or not query.strip():
            return []
        try:
            if self._embeds is None:
                self._embeds = self._l2(np.asarray(
                    self.embed_fn([c["text"] for c in self.chunks]),
                    dtype=np.float32))
            q = self._l2(np.asarray(self.embed_fn([query]), dtype=np.float32))
            sims = self._embeds @ q[0]
            order = np.argsort(sims)[::-1][:k]
            return [f'[{self.chunks[i]["source"]}] {self.chunks[i]["text"]}'
                    for i in order if sims[i] >= RETRIEVE_MIN_SIM]
        except Exception:
            return []
