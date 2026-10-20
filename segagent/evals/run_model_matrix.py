"""Evaluate Qwen3.5 9B/27B with thinking enabled and disabled.

The deterministic golden set does not contain images. Point ``--case-id`` at a
de-identified case already in the local research store, or add a ``case_id`` to
each JSONL row. The runner evaluates the actual multimodal planner rather than
substituting the rule baseline, and records reasoning availability, latency,
schema validity, action accuracy, and structure recall for each configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.knowledge import HybridKnowledgeBase  # noqa: E402
from backend.planner import QwenStructuredPlanner  # noqa: E402
from backend.schemas import PlannerAction, ToolObservation  # noqa: E402
from backend.storage import ResearchStore  # noqa: E402


DEFAULT_MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _protocol_observation(
    knowledge: HybridKnowledgeBase, question: str
) -> ToolObservation:
    match = knowledge.lookup_protocol(question)
    data: dict[str, Any] = {"matched": match is not None, "oars": []}
    summary = "No protocol matched confidently."
    if match is not None:
        data.update(
            {
                "site": match.site,
                "oars": match.oars,
                "protocol": match.model_dump(mode="json"),
            }
        )
        summary = f"Matched {match.site}: {'; '.join(match.oars)}"
    return ToolObservation(
        observation_id=f"observation_{uuid.uuid4().hex[:16]}",
        tool="lookup_protocol",
        summary=summary,
        data=data,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 1)


def evaluate_configuration(
    *,
    model_id: str,
    enable_thinking: bool,
    rows: list[dict[str, Any]],
    store: ResearchStore,
    knowledge: HybridKnowledgeBase,
    default_case_id: str | None,
    montage_slices: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    planner = QwenStructuredPlanner(
        model_id=model_id,
        store=store,
        montage_slices=montage_slices,
        max_new_tokens=max_new_tokens,
        device=device,
        enable_thinking=enable_thinking,
    )
    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    valid = 0
    first_correct = 0
    second_expected = 0
    second_correct = 0
    structure_recalls: list[float] = []
    reasoning_chars: list[int] = []

    try:
        for row in rows:
            case_id = row.get("case_id") or default_case_id
            if not case_id:
                raise ValueError(
                    f'case "{row.get("id", "?")}" has no case_id; pass --case-id '
                    "or add case_id to the JSONL row"
                )
            started = time.perf_counter()
            detail: dict[str, Any] = {
                "id": row.get("id"),
                "case_id": case_id,
                "expected_first_action": row.get("expected_first_action"),
            }
            try:
                first = planner.decide(case_id, row["question"], 1, [])
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(elapsed_ms)
                valid += 1
                first_ok = first.action.value == row.get("expected_first_action")
                first_correct += int(first_ok)
                thinking_length = len(first.thinking or "")
                reasoning_chars.append(thinking_length)
                detail.update(
                    {
                        "valid": True,
                        "latency_ms": round(elapsed_ms, 1),
                        "predicted_first_action": first.action.value,
                        "first_action_correct": first_ok,
                        "thinking_chars": thinking_length,
                    }
                )

                decision_for_structures = first
                if "expected_second_action" in row:
                    second_expected += 1
                    observations = [_protocol_observation(knowledge, row["question"])]
                    second = planner.decide(case_id, row["question"], 2, observations)
                    second_ok = (
                        second.action.value == row.get("expected_second_action")
                    )
                    second_correct += int(second_ok)
                    decision_for_structures = second
                    detail.update(
                        {
                            "predicted_second_action": second.action.value,
                            "second_action_correct": second_ok,
                            "second_thinking_chars": len(second.thinking or ""),
                        }
                    )

                expected = {
                    str(item).casefold() for item in row.get("expected_structures", [])
                }
                if expected:
                    predicted = {
                        item.casefold() for item in decision_for_structures.structures
                    }
                    recall = len(expected & predicted) / len(expected)
                    structure_recalls.append(recall)
                    detail["structure_recall"] = round(recall, 4)
                    detail["predicted_structures"] = sorted(predicted)
            except Exception as exc:
                detail.update(
                    {
                        "valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000.0, 1
                        ),
                    }
                )
            details.append(detail)
    finally:
        # Each matrix cell must release its weights before the next model loads.
        del planner
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    total = len(rows)
    return {
        "model": model_id,
        "thinking_enabled": enable_thinking,
        "cases": total,
        "valid_decision_rate": round(valid / max(total, 1), 4),
        "first_action_accuracy": round(first_correct / max(total, 1), 4),
        "second_action_accuracy": round(
            second_correct / max(second_expected, 1), 4
        ),
        "mean_structure_recall": round(
            statistics.fmean(structure_recalls), 4
        )
        if structure_recalls
        else None,
        "thinking_presence_rate": round(
            sum(value > 0 for value in reasoning_chars) / max(valid, 1), 4
        ),
        "mean_thinking_chars": round(statistics.fmean(reasoning_chars), 1)
        if reasoning_chars
        else 0.0,
        "mean_latency_ms": round(statistics.fmean(latencies), 1)
        if latencies
        else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "evals" / "golden_cases.jsonl"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT / ".research_data")
    parser.add_argument(
        "--case-id",
        help="Use one de-identified stored case for rows that do not specify case_id.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--thinking",
        choices=("both", "on", "off"),
        default="both",
        help="Evaluate both thinking settings, or only one.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--montage-slices", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset = _rows(args.dataset)
    store = ResearchStore(args.data_dir)
    knowledge = HybridKnowledgeBase(PROJECT / "knowledge", embed_fn=None)
    thinking_modes = {
        "both": (True, False),
        "on": (True,),
        "off": (False,),
    }[args.thinking]
    results = [
        evaluate_configuration(
            model_id=model_id,
            enable_thinking=enabled,
            rows=dataset,
            store=store,
            knowledge=knowledge,
            default_case_id=args.case_id,
            montage_slices=args.montage_slices,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        for model_id in args.models
        for enabled in thinking_modes
    ]
    payload = {
        "dataset": str(args.dataset),
        "data_dir": str(args.data_dir),
        "matrix": results,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
