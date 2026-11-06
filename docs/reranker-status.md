# Reranker reconciliation — 6 November 2026

This branch brings the existing reranker stack together with merged team main
`d11a5ea32cc1f213001b378d048c2c0550e7130b`. Main contributes split configuration,
preprocessing, BERT training/prediction and shared evaluation. Its twelve added
files are retained byte-for-byte; no real-data training, BERT inference or
supplied-evaluator execution is part of this reconciliation. The common ancestor was setup `c9780753`:
the reranker branch had six commits and main had seventeen commits after it.
The separate open teammate PR #28 is not merged here.

## PR disposition and evidence

Local Git ancestry and each PR's actual file diff were inspected. The following
closures are from the coordinating reviewer's GitHub readback on 6 November;
closed does not mean merged into main. Historical branches and pilot evidence
remain available.

| PR / head | Disposition on this branch |
| --- | --- |
| #19 / `6c293edf` | Already inherited: numerical ridge repair, core tests and notes (3 files). Closed unmerged as superseded; no duplicate merge. |
| #20 / `c4055c9d` | Already inherited: provisional offline handoff contract, synthetic fixtures, tests and documentation (6 files). Closed unmerged as superseded. |
| #21 / `8733c6d9` | Already inherited: synthetic integration and evaluator-boundary checks (7 files). Closed unmerged as superseded. |
| #22 / `430b450c` | Already inherited: label-free held-out synthetic inference (4 files). Closed unmerged as superseded. |
| #23 / `fd2a0c03` | Excluded, closed without merge: historical HF/seed-42 80/10/10 preparation and its producer-specific receipt verifier (5 files, 2 commits). |

The excluded files are `squad_preparation.py`, `preparation_receipt.py`, their two
test files and `docs/realdata-preparation.md`. Their producer hashes, assignments
and receipts describe a different source/split policy; they cannot certify the
team's Canvas/seed-4248 75/15/10 split by changing ratios or seed alone.
The active team's policy is in `configs/split.json` and `scripts/split_data.py`.
Its published question counts are 65,952 / 12,516 / 9,131, not counts recomputed
by this reconciliation.

## Which interface to use

| Path | Current purpose and boundary |
| --- | --- |
| `reranker/team_candidates.py` | Active schema-version-1 adapter for inspected PR28 `e56efaa1aacfa5d547286137375ab398baaeefeb`; preserves supplied candidate order, ties, span identities and float32/float64 score semantics. |
| `reranker/team_model.py` | Implemented `TeamCandidateModel` synthetic fit/save/load/new-batch prediction wrapper, binding exact feature order/spec/group, declared generation metadata and training membership. Reload does not fit. |
| `reranker/core.py` and `python -m reranker` | Generic numerical core/CLI. Caller-supplied features and labels do not establish provenance or an approved real-data policy. |
| `offline_handoff/`, `reranker/integration.py`, `reranker/heldout.py` | Retained older draft synthetic bundle contracts and replay/held-out tests. These do not consume PR28 exports directly. |
| `scripts/squad_eval.py`, `evaluate-v2.0.py` | Team's shared evaluator and checked-in source, integrated from main. Not replaced by the synthetic evaluator boundary. |
| `reranker/evaluation.py` | Complete-ID preflight and injectable transport with synthetic test doubles; no default evaluator implementation or executable conformance claim. |

The active adapter derives a provisional basic-five/full-fifteen feature registry.
All question-type encodings and interactions are full-only. Labels remain a
separate exact-coverage mapping. Caller-supplied feature keys and explicit
gold/target fields are rejected; the adapter cannot authenticate the origin of
supplied logits. Neither feature acceptance, F1 target policy nor no-signal methodology is team-agreed by
this code. Synthetic wrappers require explicit hand-authored-target status and
must not be relabelled as a real-data training path.

`TeamCandidateModel` reload needs the original label-free training export and an
independently retained model-file SHA-256. New compatible internal_val batches
have their own hashes and may contain unseen IDs/text; training overlap is rejected.
Checkpoint paths and inspected producer revisions are declarations, not proof
of the exact checkpoint/tokenizer bytes used. Hashes establish byte identity
relative to a trusted pin, not origin or accepted split membership.

## Current issues and remaining inputs

The issue state below was supplied by the coordinating reviewer's fresh GitHub
reads on 6 November. Hai delivered #2–#6 (closed); Sean owns #7/PR28; Xu An owns
#10. Assignment is not evidence of completion. No issue is closed by this commit.

| Issue | Remaining scope |
| --- | --- |
| #8, open/unassigned | Depends on #7: agree/document score, token-length, text and removable question-type features; validate consistent basic/full vectors on actual exports. Our feature registry is provisional. |
| #9, open/unassigned | Depends on #7/#8: supply separate training candidates/features and gold supervision, target/objective/no-useful-candidate policy, with validation separate. The 21 October request for scorer provenance and separate fixtures has no teammate reply. PR28 exports omit features/golds/targets. |
| #10, open/Xateh | Depends on #9: train both variants on real reranker-training inputs, freeze train-only preprocessing, save/reload model/config and produce shared-format predictions. Synthetic coverage does not finish this issue. |
| #11, open/unassigned | Depends on #10: compare baseline/basic/full on identical internal validation and scorer; record seeds, settings, runtime and EM/F1, then freeze configurations without consulting dev. |
| #12, open/unassigned | Depends on #11: report all three frozen systems on official dev with artifact/config references, without tuning. No complete three-system result is established here. |
| #13, open/unassigned | Depends on #11/#12: candidate coverage/oracle analysis, corrections/regressions, question-type breakdown and representative errors. No new analysis results are claimed. |

Main now contains evaluator bytes, so saying that no evaluator exists is obsolete.
Their byte identity can be recorded locally; correspondence to the exact supplied
course download and an agreed label-generation invocation remain separate
provenance checks. No denied source retrieval is needed or attempted here.

Team `docs/eval.md` reports baseline internal-validation 65.40 EM / 79.29 F1 and
dev 80.02 / 87.55; these are team-reported results, not reruns or reranker gains.
Official dev is therefore not untouched. Main's documented baseline uses
max-context filtering, while the inspected PR28 export profile compares all
windows without it. Obtain matching baseline mappings for the same export
configuration; do not assume the published metrics validate PR28's profile.

The next real-input gate needs actual reranker_train/internal_val candidate
exports and matching baseline mappings, exact team split membership and source
identities, checkpoint/tokenizer/config hashes and accepted feature/target policy.
Check full supplied split metadata for question/article/exact-context overlap
before sampling. A small schema or baseline-alignment sample cannot certify
whole-dataset coverage or leakage safety. These artifacts have not been established
as available in the inspected xlogin project locations; that is not evidence
that teammates have no unpublished artifacts elsewhere.

## Bounded local regression commands

Run from the repository root on `issue10-team-adapter-20261105`; a default-main
checkout does not contain the reranker stack. Use the existing pinned Python
environment, one CPU/thread, CUDA disabled and
the task-specific memory/runtime limits. Each suite runs in a separate process,
because the offline contract suite checks that numerical/model modules have not
been imported. These tests use invented inputs; the evaluator-boundary suite
uses a test double and does not execute `evaluate-v2.0.py`.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
for suite in test_reranker.py test_offline_handoff_contract.py \
  test_handoff_integration.py test_evaluator_boundary.py \
  test_heldout_inference.py test_team_candidates.py test_team_model.py
do
  python -m unittest discover -s tests -p "$suite" -v || exit 1
done
```

For label-free PR28 conversion (only after actual input/provenance preflight):

```bash
python -m reranker.team_candidates --input internal_val_candidates.json \
  --baseline internal_val_predictions.json --group full --output adapted-full.json
```

Use a new output path. This performs structural conversion and exact baseline
answer/ID matching, not model fitting, prediction, label generation or evaluation.
For tiny synthetic model save/reload, see [TeamCandidateModel](team-model.md).
A real-input training command is intentionally not claimed ready while #8/#9
policy and artifacts remain unresolved.

On 6 November, after integrating the pinned main tree, these seven suites passed
94/94 tests (18 core, 31 offline contract, 9 synthetic integration, 5 evaluator
boundary, 10 held-out, 12 team adapter and 9 team model). Each ran in a separate
process using the existing Python environment. Total measured execution including
launcher/import overhead was 23.911 seconds, on one CPU with a 1 GiB address-space
cap and CUDA disabled; maximum child RSS was 220,764 KiB. No new dependency,
real-data run or official evaluator execution was involved. This is local
synthetic regression evidence, not CI or real QA performance evidence.
