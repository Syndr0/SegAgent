# Evaluation and proposed research protocol

This document distinguishes the evaluation implemented in this repository from
experiments that require additional datasets, instrumentation, or code. It does
not report benchmark results. SegAgent is research software; passing its tests
or text fixtures does not establish segmentation quality or clinical validity.

## Implemented evaluation

[`evals/run_evals.py`](evals/run_evals.py) evaluates `RulePlanner` against
[`evals/golden_cases.jsonl`](evals/golden_cases.jsonl). The bundled dataset contains
six text fixtures: explicit structure requests, protocol-driven structure
selection, a QC request, and an ambiguous request.

The runner constructs `HybridKnowledgeBase` with `embed_fn=None`. Protocol
lookup uses its alias matching and lexical BM25 fallback without semantic
embeddings. It asks the rule planner for a first decision and, only after a
protocol-lookup decision, feeds the lookup result back for a second decision.

The runner does not load Qwen or VoxTell, read image volumes, execute segmentation
or QC, run the durable workflow, or exercise human review. A predicted `segment`
or `run_qc` action is a routing decision, not a completed tool invocation.

This retrieval configuration is specific to the evaluation script. The runtime
service supplies an embedding function to its knowledge base and can use hybrid
retrieval even when the selected runtime planner is `rule`. Do not label the
runtime rule planner as inherently BM25-only.

### Run the fixture evaluation

With the project dependencies installed, run from `segagent-research-v2/`:

```bash
python evals/run_evals.py --dataset evals/golden_cases.jsonl
```

The command prints a JSON report with aggregate scores and per-fixture details.
It accepts an alternative JSONL dataset through `--dataset`. The command shown
here is an instruction, not a claim that an evaluation has been executed.

### Reported metrics and limitations

| JSON field | Definition | Bundled fixture coverage |
| --- | --- | --- |
| `first_action_accuracy` | Fraction of all fixtures whose first action matches the expected action. | 6 fixtures. |
| `second_action_accuracy` | Fraction of fixtures declaring an expected second action whose second action matches. A missing second decision is incorrect. | 2 protocol fixtures. |
| `protocol_site_accuracy` | Fraction of fixtures declaring an expected site whose lookup returns that site. | 2 protocol fixtures. |
| `mean_structure_recall` | Mean per-fixture recall of case-insensitive expected structure sets, using the second decision when present and otherwise the first. Only fixtures with nonempty expected sets contribute. | 4 structure fixtures. |
| `abstention_accuracy` | Fraction of all fixtures for which the first action being `ask_user` agrees with `expect_abstention`. | 6 fixtures, including 1 expected abstention. |

These are checks of a small, hand-authored routing fixture set. They are not
held-out model benchmarks. Structure recall does not penalize extra predicted
structures, and abstention accuracy is not abstention precision or recall.
First- and second-action agreement does not measure a complete trajectory.

The report does not measure invalid model decisions, image understanding,
citation correctness, final-answer support, mask overlap, clinical utility,
latency, reviewer behavior, or recovery after interruption. If a replacement
dataset has no eligible fixtures for a conditional metric, the script's guarded
division returns zero; report the eligible count and mark that metric as
unassessed rather than interpreting the zero as measured performance.

## Research questions for future experiments

The following questions are proposed research directions, not conclusions:

1. Does typed planning reduce invalid decisions and unsupported tool requests
   compared with a free-form planner under matched model and tool conditions?
2. Does protocol retrieval improve treatment-site selection, requested-structure
   coverage, and source attribution on held-out requests?
3. Does durable state preserve case isolation and correct execution when a run
   is interrupted, reviewed, and resumed?
4. How does human review affect mask acceptance, reviewer effort, and the use of
   rejected artifacts in subsequent answers?
5. Does an expert-mask cross-check improve contour-error detection relative to
   deterministic QC alone, and what false-alert burden does it introduce?

## Proposed comparisons and ablations

None of the comparisons below has a result reported in this document. Existing
runtime settings provide some conditions, but the fixture runner does not
implement a multi-system benchmark.

| Proposed condition | Current support | Work required for a controlled comparison |
| --- | --- | --- |
| Rule versus Qwen planning | Runtime selection through `SEGAGENT_PLANNER=rule` or `qwen`. | Add an evaluation adapter for Qwen and matched observations; the existing runner evaluates only rules. |
| Legacy versus v2 workflow | Both implementations are present in separate directories. | Define shared cases, equivalent tool/model settings, supported tasks, and output scoring. Do not assume feature parity. |
| Lexical versus hybrid retrieval | The fixture runner disables embeddings; the runtime service supplies an embedding function. | Add an explicit comparison harness with fixed knowledge snapshots and matched planner conditions. |
| Retrieval disabled | No dedicated no-RAG runtime switch. | Implement an isolated ablation and define how the planner behaves without retrieved protocols or passages. |
| Human review enabled versus disabled | `SEGAGENT_REQUIRE_MASK_APPROVAL=true` or `false` controls the runtime segmentation review step. | Add workflow-level evaluation and reviewer labels. The disabled condition automatically approves that step; record it as an experimental configuration. |
| Deterministic QC versus expert-mask cross-check | No dedicated no-expert runtime switch. | Implement a controlled QC ablation and score it against independently annotated contour issues. |
| Durable resume versus fresh execution | Checkpointed workflow code is present. | Add interruption and recovery scenarios, instrumentation, and assertions on state and artifacts. |

An ablation should vary one intended factor while holding models, knowledge,
inputs, tool availability, and scoring fixed. Document code changes needed to
create each condition instead of presenting proposed conditions as available
configuration flags.

## Proposed image and workflow evaluation

These measurements require additional data and harnesses; they are not outputs
of `run_evals.py`:

- **Segmentation:** Dice, surface Dice with a declared tolerance, HD95 in physical
  units, volume error, and empty-mask or tool-failure rates. Define behavior for
  absent structures and empty reference or predicted masks before scoring.
- **Retrieval and answers:** protocol-site accuracy, structure precision and
  recall, citation correctness, and unsupported-claim rate, using independent
  labels and a written adjudication rubric.
- **Contour QC:** issue-level precision and recall, false alerts per case,
  agreement with expert annotations, and reviewer time. Keep reference labels
  independent of the model used for expert-mask comparison.
- **Planning:** complete-trajectory success, invalid-decision rate, clarification
  precision and recall, tool failures, and steps to completion.
- **Reliability and resources:** resume success, duplicate tool execution,
  approval handling, cross-case artifact isolation, event completeness,
  latency, peak GPU memory, and compute cost per completed case.

Before computing segmentation metrics, verify orientation, affine transforms,
voxel spacing, dimensions, label mapping, and any resampling. Use the same
geometry policy across conditions and retain its provenance. Report per-case
and per-structure results, denominators, failures, and exclusions; aggregate
means alone can hide important failure modes.

## Reproducibility and reporting protocol

1. **Freeze the experiment.** Record the repository commit, configuration,
   dependency versions, exact command, and any uncommitted experimental patch.
   Identify the planner, segmentation model, text encoder, and knowledge sources
   by immutable revision or checksum where available.
2. **Define the dataset.** Record dataset version, provenance, inclusion criteria,
   labels, train/development/test split, and any failure-focused subset. Split
   by patient or another justified independent unit to avoid leakage. Keep
   development fixtures separate from claims of held-out generalization.
3. **Control execution.** Record random seeds, decoding and sampling settings,
   hardware, OS, Python, GPU driver/CUDA, precision, and relevant environment
   settings. Document nondeterminism and repeated-run policy; a fixed seed alone
   does not guarantee identical model behavior.
4. **Predefine scoring.** Specify metric implementations, units, tolerances,
   eligibility rules, missing-value handling, aggregation, and annotation
   procedure. Report uncertainty only with an appropriate sample and method;
   the six bundled fixtures do not support broad performance claims.
5. **Audit recovery and review.** Test interruption before and after tool calls,
   while awaiting approval, and across process restart. Check that resumption
   uses the intended case/run, preserves artifact references, handles duplicate
   requests, and records approve, reject, and feedback decisions correctly.
   Investigate unintended approval bypasses and cross-case access separately
   from ordinary task failures.
6. **Publish evidence and limits.** Include aggregate scores, denominators,
   failure categories, resource measurements, and the exact experimental
   conditions. Release only appropriately licensed, synthetic or de-identified
   artifacts. Do not publish protected health information, credentials, private
   traces, or downloaded weights without redistribution rights.

Report measured findings only after completing the corresponding experiment.
Describe unimplemented conditions as proposals, and do not infer clinical
validation from routing fixtures, unit tests, or model agreement.
