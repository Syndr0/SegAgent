# Evaluation and research protocol

This document describes evaluation support in **SegAgent**, implemented in
`segagent/`. It reports no benchmark results; proposed experiments are identified
separately from the runners and runtime measurements already implemented.

## Implemented evaluation runners

### Deterministic routing fixtures

[`evals/run_evals.py`](evals/run_evals.py) evaluates `RulePlanner` against
[`evals/golden_cases.jsonl`](evals/golden_cases.jsonl). The bundled dataset
contains six text fixtures: explicit structure requests, protocol-driven
structure selection, a QC request, and an ambiguous request.

The runner constructs `HybridKnowledgeBase` with `embed_fn=None`. Protocol
lookup uses alias matching and lexical BM25 fallback without semantic
embeddings. It evaluates a first decision and, after a protocol-lookup action,
feeds the lookup result back for a second decision.

With the project dependencies installed, run from `segagent/`:

```bash
python evals/run_evals.py --dataset evals/golden_cases.jsonl
```

The runner prints a JSON report containing aggregate scores and fixture details:

| JSON field | Definition | Bundled coverage |
| --- | --- | --- |
| `first_action_accuracy` | Fraction of all fixtures whose first action matches the expected action. | 6 fixtures. |
| `second_action_accuracy` | Fraction of fixtures declaring an expected second action whose second action matches; a missing second decision is incorrect. | 2 protocol fixtures. |
| `protocol_site_accuracy` | Fraction of fixtures declaring an expected site whose lookup returns that site. | 2 protocol fixtures. |
| `mean_structure_recall` | Mean per-fixture recall of case-insensitive expected structure sets, using the second decision when present and otherwise the first. | 4 fixtures with nonempty expected sets. |
| `abstention_accuracy` | Fraction of all fixtures for which the first action being `ask_user` agrees with `expect_abstention`. | 6 fixtures, including 1 expected abstention. |

This runner loads neither Qwen nor VoxTell, reads no image volumes, and invokes
neither segmentation nor QC. A predicted `segment` or `run_qc` action is a
routing decision, not a completed tool call. The runner also does not exercise
the dialogue workflow, clarification gates, final-answer verifier, or recovery.

These six hand-authored fixtures are a regression aid, not a held-out benchmark.
Structure recall does not penalize extra structures; abstention accuracy is
not abstention precision or recall. If an alternative dataset has no eligible
fixtures for a conditional metric, the guarded division returns zero. Report
the eligible count and mark that metric unassessed.

The runtime service supplies an embedding function to its knowledge base,
including when `SEGAGENT_PLANNER=rule`. The fixture runner's lexical retrieval
configuration does not characterize all rule-planner runtime requests.

### Multimodal planner model matrix

[`evals/run_model_matrix.py`](evals/run_model_matrix.py) invokes
`QwenStructuredPlanner` with image views and local case artifacts. Its default
matrix contains four configurations:

| Planner checkpoint | Thinking requested |
| --- | --- |
| `Qwen/Qwen3.5-9B` | Enabled |
| `Qwen/Qwen3.5-9B` | Disabled |
| `Qwen/Qwen3.5-27B` | Enabled |
| `Qwen/Qwen3.5-27B` | Disabled |

The models are evaluation conditions, not separate application versions.
Qwen3.5-9B is the runtime default; the 27B checkpoint is an evaluation candidate.

The runner requires an existing, appropriately de-identified case in the local
research store. Each JSONL row may specify `case_id`; `--case-id` supplies the
fallback for rows without one. Replace the example ID below with a real stored
case ID before running:

```bash
python evals/run_model_matrix.py \
  --dataset evals/golden_cases.jsonl \
  --data-dir .research_data \
  --case-id case_REPLACE_WITH_STORED_ID \
  --thinking both \
  --output .research_data/model-matrix.json
```

The runner prints JSON and optionally writes the same payload through
`--output`. The parent directory of that output file must already exist. Reusing one image
for every bundled text fixture is useful for an execution check, but does not
create a representative image benchmark. For meaningful evaluation, author
case-specific questions and labels, then freeze a held-out split.

The matrix reports:

- `valid_decision_rate`: first planner calls returning a validated decision
  divided by all rows. The planner may repair an invalid response once, so this
  is not raw first-generation schema validity.
- `first_action_accuracy` and `second_action_accuracy`: action agreement with
  fixture labels. For rows declaring a second action, the runner supplies a
  protocol observation and requests a second decision regardless of the first
  action. This is a controlled follow-up, not a complete autonomous trajectory.
- `mean_structure_recall`: mean recall over rows whose expected structure sets
  reach the scoring step. Failed rows can be absent from this mean; review
  per-row errors and report the intended and scored counts.
- `thinking_presence_rate` and `mean_thinking_chars`: availability and length
  of the first decision's captured thinking. Presence or length does not
  establish reasoning correctness, usefulness, or clinical evidence.
- `mean_latency_ms` and `p95_latency_ms`: elapsed time for successful first
  planner calls. The first call includes lazy model loading; retries and image
  preparation also contribute. These are not warmed generation-only or
  end-to-end workflow timings.

Inspect `details` when interpreting aggregates. A first call can count toward
`valid_decision_rate` even if its second call later fails; the row detail then
records the exception. The second-action denominator includes only rows that
reached that stage after a successful first call. Small-sample percentile
estimates and conditional averages should not be treated as broad performance
claims.

Both runners use lexical protocol lookup without embeddings. The model matrix
does not execute VoxTell, score masks, measure QC accuracy, exercise multi-turn
workflow recovery, or verify final-answer grounding. It constructs planners
directly; it does not apply every runtime `Settings` option or reproduce the
runtime's intent-gating path. Record its explicit CLI arguments and planner
defaults, especially image budget, device placement, and token budget. Thinking
can consume the generation budget before JSON, and the planner's repair attempt
disables thinking; record this behavior when comparing conditions.

The commands above describe evaluation entry points. Their presence does not
claim that they have been executed or that any configuration has passed.

## Runtime evidence and QC measurements

The runtime already produces structured evidence, but neither runner above
constitutes a benchmark of that evidence pipeline:

- `MetricEvidence` carries the measured quantity, unit, structure, source, and
  artifact references. QC/report evidence is used by the deterministic final
  answer verifier to check numeric claims and citations. Heuristic verification
  is not a guarantee that every factual claim is correct.
- Contour QC compares recognized organ/OAR contours with VoxTell masks. Where
  available, it records Dice, contour/model volumes, volume difference, mean
  surface distance, and HD95, together with comparison artifacts. These are
  agreement measurements against a model prediction, not accuracy against an
  independent reference annotation.
- Targets such as GTV, CTV, and PTV receive descriptive measurements rather than
  anatomy-model correctness judgments. Auxiliary and unclassified structures
  have separate handling. Evaluate role classification and exclusions explicitly.
- Protocol completeness depends on applicability checks. An unassessed protocol
  cannot support a claim that required anatomy is missing.
- Mask-aware follow-up answers, artifact references, prompt normalization,
  dialogue continuity, clarification gates, and finalization guards have code
  and test coverage. Their presence does not replace measured model or workflow
  performance on representative cases.

Use independent reference annotations and a documented scoring protocol before
reporting segmentation or contour-QC accuracy. Keep the distinction between
model agreement, deterministic checks, and expert adjudication in all results.

## Proposed research questions and comparisons

The following are research questions, not reported findings:

1. How do the 9B and 27B planners compare under matched inputs, token budgets,
   and thinking settings, including schema-repair and failure behavior?
2. Does typed planning reduce invalid tool requests compared with a matched
   free-form planner?
3. Does protocol retrieval improve requested-structure selection and source
   attribution on held-out questions?
4. Does mask context improve case-bound follow-up answers and artifact-reference
   resolution without increasing unsupported claims?
5. Does expert-mask comparison improve contour-error detection relative to
   deterministic QC alone, and at what false-alert burden?
6. Do clarification gates and durable dialogue preserve correct execution across
   interrupted streams, process restarts, and ambiguous follow-up requests?

| Proposed condition | Existing support | Additional evaluation work |
| --- | --- | --- |
| Qwen 9B/27B, thinking enabled/disabled | Implemented four-cell planner runner. | Add representative case-specific labels, failure-aware scoring, and controlled resource measurements. |
| Rule versus Qwen planning | Runtime `SEGAGENT_PLANNER=rule` or `qwen`; separate runners. | Match inputs, observations, retrieval, and scoring before comparing their reports. |
| Lexical versus hybrid retrieval | Lexical runners and embedding-backed runtime knowledge base. | Add one matched retrieval harness with a frozen knowledge snapshot. |
| Retrieval disabled | No dedicated no-RAG runtime switch. | Implement an isolated ablation and specify planner behavior without retrieved knowledge. |
| Mask context disabled | No dedicated no-mask-context ablation switch. | Control selected artifact views and references without changing other planner inputs. |
| Deterministic QC without expert comparison | No dedicated no-expert runtime switch. | Isolate the comparison component and use independent issue annotations. |
| Interrupted versus uninterrupted execution | Checkpointed dialogue and clarification gates. | Add interruption schedules, restart scenarios, state assertions, and event auditing. |

The current configuration has no `SEGAGENT_REQUIRE_MASK_APPROVAL` setting.
Its clarification gates should not be described as the earlier blanket mask
approval step. Any future human-review study must define its intervention and
implementation before evaluation.

For each ablation, vary one intended factor and document the required patch.
Do not present proposed conditions as ready-made configuration flags.

## Proposed image and workflow benchmark

Additional datasets and instrumentation are needed for these measurements:

- **Segmentation:** Dice, surface Dice with a declared tolerance, HD95 in
  physical units, volume error, and empty-mask or tool-failure rates against
  independent references. Predefine absent-structure and empty-mask handling.
- **Answers and retrieval:** structure precision and recall, citation correctness,
  unsupported-claim rate, and artifact-reference accuracy with an independent
  annotation and adjudication rubric.
- **Contour QC:** issue-level precision and recall, false alerts per case,
  expert agreement, and reviewer effort. Reference labels must be independent
  of the model used for comparison.
- **Thinking visibility:** stream completeness, truncation, time to first
  displayed token, and human-rated usefulness. Raw thinking is not admitted
  measurement evidence.
- **Workflow reliability:** task completion, duplicate tool calls, clarification
  handling, cross-case isolation, stale artifact references, event completeness,
  and resume success.
- **Resources:** warm and cold latency, peak memory, tool runtime, and compute
  cost per completed case, with failed cases retained in the accounting.

Before scoring masks, verify orientation, affine transforms, voxel spacing,
dimensions, labels, and resampling. Use the same geometry policy across
conditions and preserve its provenance. Report case- and structure-level
denominators, failures, and exclusions alongside aggregate scores.

## Reproducibility and reporting

1. **Freeze the experiment.** Record the repository commit, any experimental
   patch, dependency versions, configuration, and exact command. Identify planner,
   segmentation model, text encoder, and knowledge sources by immutable revision
   or checksum. Local snapshot paths may be needed to pin model revisions.
2. **Define the dataset.** Record provenance, inclusion criteria, annotation
   procedure, dataset version, splits, and a separate failure-focused subset.
   Split by patient or another justified independent unit to prevent leakage.
   Keep development fixtures separate from held-out claims.
3. **Control execution.** Record random seeds, deterministic decoding settings,
   hardware, OS, Python, GPU driver/CUDA, precision, device placement, image
   budget, and generation limits. Document nondeterminism and repeated-run
   policy; a fixed seed alone does not guarantee identical model behavior.
4. **Predefine scoring.** Specify implementations, units, tolerances, aggregation,
   eligibility, missing-value handling, and uncertainty methods. Report intended
   and successfully scored counts. Six fixtures cannot establish general model
   performance.
5. **Audit recovery.** Interrupt before and after tool execution, during a
   clarification gate, and across process restart. Verify case/run identity,
   conversation state, artifact provenance, duplicate-request handling, and
   reference resolution. Audit cross-case access and unsupported final claims.
6. **Publish evidence and limits.** Include aggregate scores, failure categories,
   resource measurements, and exact conditions. Release only appropriately
   licensed, synthetic or de-identified artifacts. Exclude PHI, credentials,
   private traces, and weights without redistribution permission.

Report improvements only after running the corresponding experiment. Routing
fixtures, unit tests, model agreement, and verifier acceptance do not establish
clinical validity.
