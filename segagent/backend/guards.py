from __future__ import annotations

import hashlib
from typing import Iterable


def request_signature(
    case_id: str, tool: str, structures: Iterable[str], model_version: str = ""
) -> str:
    """Stable signature for a tool request.

    Order-, case-, and whitespace-insensitive over the structure list, so an
    identical re-issued request maps to the same key regardless of phrasing.
    """
    normalized = tuple(
        sorted(" ".join(str(item).strip().split()).casefold() for item in structures)
    )
    payload = repr((case_id, tool, normalized, model_version))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class SeenRequests:
    """Idempotency guard preventing a byte-identical tool call from re-running.

    VoxTell is deterministic given (image, prompt), so re-issuing the same
    segmentation cannot produce a different mask. This turns a futile retry
    loop into a signal the planner must act on (refine the request or stop).
    """

    def __init__(self, signatures: Iterable[str] | None = None):
        self._seen: set[str] = set(signatures or [])

    def is_new(self, signature: str) -> bool:
        if signature in self._seen:
            return False
        self._seen.add(signature)
        return True

    def seen(self, signature: str) -> bool:
        return signature in self._seen
