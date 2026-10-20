from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from backend.knowledge import HybridKnowledgeBase  # noqa: E402
from backend.planner import RulePlanner  # noqa: E402
from backend.schemas import PlannerAction, ToolObservation  # noqa: E402


@dataclass
class Score:
    total: int = 0
    first_action_correct: int = 0
    second_action_correct: int = 0
    second_action_expected: int = 0
    site_correct: int = 0
    site_expected: int = 0
    structure_recall_sum: float = 0.0
    structure_cases: int = 0
    abstention_correct: int = 0
    details: list[dict] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "cases": self.total,
            "first_action_accuracy": round(self.first_action_correct / max(self.total, 1), 4),
            "second_action_accuracy": round(
                self.second_action_correct / max(self.second_action_expected, 1), 4
            ),
            "protocol_site_accuracy": round(self.site_correct / max(self.site_expected, 1), 4),
            "mean_structure_recall": round(
                self.structure_recall_sum / max(self.structure_cases, 1), 4
            ),
            "abstention_accuracy": round(self.abstention_correct / max(self.total, 1), 4),
            "details": self.details,
        }


def evaluate(dataset: Path) -> dict:
    knowledge = HybridKnowledgeBase(PROJECT / "knowledge", embed_fn=None)
    planner = RulePlanner(knowledge)
    score = Score()
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        score.total += 1
        observations = []
        first = planner.decide("case_000000000000", case["question"], 1, observations)
        first_ok = first.action.value == case["expected_first_action"]
        score.first_action_correct += int(first_ok)
        predicted_site = None
        second = None
        if first.action == PlannerAction.LOOKUP_PROTOCOL:
            match = knowledge.lookup_protocol(first.site_query or case["question"])
            if match is None:
                summary = "No protocol matched confidently."
                data = {"matched": False, "oars": []}
            else:
                summary = f"Matched {match.site}: {'; '.join(match.oars)}"
                data = {
                    "matched": True,
                    "site": match.site,
                    "oars": match.oars,
                    "protocol": match.model_dump(mode="json"),
                }
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="lookup_protocol",
                summary=summary,
                data=data,
            )
            observations.append(observation)
            predicted_site = match.site if match else None
            second = planner.decide("case_000000000000", case["question"], 2, observations)
        if "expected_site" in case:
            score.site_expected += 1
            score.site_correct += int(predicted_site == case["expected_site"])
        if "expected_second_action" in case:
            score.second_action_expected += 1
            score.second_action_correct += int(
                second is not None and second.action.value == case["expected_second_action"]
            )
        decision_for_structures = second or first
        expected = {item.casefold() for item in case.get("expected_structures", [])}
        predicted = {item.casefold() for item in decision_for_structures.structures}
        recall = len(expected & predicted) / max(len(expected), 1) if expected else 1.0
        if expected:
            score.structure_cases += 1
            score.structure_recall_sum += recall
        abstained = first.action == PlannerAction.ASK_USER
        score.abstention_correct += int(abstained == bool(case.get("expect_abstention")))
        score.details.append(
            {
                **case,
                "predicted_first_action": first.action.value,
                "predicted_second_action": second.action.value if second else None,
                "predicted_site": predicted_site,
                "predicted_structures": sorted(predicted),
                "structure_recall": recall,
                "first_action_correct": first_ok,
            }
        )
    return score.report()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "evals" / "golden_cases.jsonl"
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset), indent=2))


if __name__ == "__main__":
    main()
