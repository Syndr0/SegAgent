# SegAgent Research Experiments

The project is structured as a research artifact, not a clinical product. Every
reported result should include the dataset version, model commit or model ID,
planner configuration, random seed, hardware, and the exact evaluation command.

## Research questions

1. Does typed planning reduce invalid or hallucinated tool calls compared with
   free-form action parsing?
2. Does a durable graph improve recovery after interruption and human review?
3. Does protocol retrieval improve OAR selection accuracy and source attribution?
4. Does human approval prevent low-quality masks from entering the final answer?
5. Does the expert cross-check find more contour problems than deterministic QC
   alone without creating an unacceptable false-positive burden?

## Baselines and ablations

| System | Typed planner | Protocol retrieval | Durable graph | Human review | Expert QC |
|---|---:|---:|---:|---:|---:|
| Original prototype baseline | No | Keyword only | No | No | Partial |
| Rule baseline | Yes | BM25 | Yes | Yes | Yes |
| Qwen planner | Yes | Hybrid | Yes | Yes | Yes |
| No-RAG ablation | Yes | No | Yes | Yes | Yes |
| No-HITL ablation | Yes | Hybrid | Yes | No | Yes |
| Deterministic-QC ablation | Yes | Hybrid | Yes | Yes | No |

## Metrics

- Planning: first-action accuracy, trajectory accuracy, invalid-decision rate,
  clarification/abstention precision and recall, and step count.
- Retrieval: protocol-site accuracy, structure recall, citation precision, recall,
  and unsupported-claim rate.
- Segmentation: Dice, normalized surface Dice, Hausdorff 95, volume error, latency,
  and failure/empty-mask rate, stratified by structure and site.
- QC: issue-level precision/recall, false alerts per case, expert agreement, and
  reviewer time.
- Agent reliability: resume success, approval bypass rate, cross-case artifact
  leakage, event completeness, and cost or GPU time per completed case.

## Evaluation protocol

1. Freeze a de-identified case split and keep a separate failure-focused set.
2. Run the deterministic golden trajectory suite with `python evals/run_evals.py`.
3. Run unit tests with `pytest -q`.
4. Evaluate each system on identical cases and tool/model versions.
5. Bootstrap confidence intervals at the case level and report per-organ results.
6. Manually audit every unsupported claim and every HITL bypass failure.
7. Publish configs and aggregate results; never publish protected health data.

## Resume-ready evidence

Good resume bullets should contain measured outcomes, for example: "Built a
LangGraph-based multimodal medical-imaging agent with typed tool contracts,
checkpointed HITL review, case-isolated artifacts, hybrid protocol retrieval,
and OpenTelemetry traces; improved held-out trajectory accuracy from X to Y and
reduced unsupported final-answer claims from A% to B%." Replace every placeholder
with a reproducible result. Do not claim clinical validation from research tests.
