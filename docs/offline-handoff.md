# Offline handoff validator — provisional software contract

This implements the supplied 21 October review draft as a **synthetic software
boundary**, not team agreement, scorer verification or methodology acceptance.
It preserves contract ID `cs4248.reranker-handoff/0.1.0-draft` and feature spec
`candidate-features/0.1.0-draft`. The additional runnable bundle schema is
`cs4248.offline-bundle/0.1.0-draft`. Future draft changes require a new explicit
ID. After 1.0, incompatible semantics/order/offset/label/split changes require a
major version; additive metadata a minor version; editorial changes a patch.
The core pin is `6c293edf63a0816370c1407a880b33c3ff2a6176` (reviewed PR19).

The new `offline_handoff` package uses only the Python standard library and never
imports the reranker, fits a model, predicts, tokenizes, or computes labels. The
existing core and earlier handoff work are unchanged. The proposed raw-logit
comparison across windows, feature choices, token-F1 target and centered-ridge
objective still require owner review. This module does not establish that they
are scientifically appropriate or reproduce the baseline generator.

## Runnable synthetic boundary

From this worktree, using the existing user-home CPU interpreter:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
PY=/home/t/tehxa/cs4248-issue10-20261017/repo/.venv/bin/python
$PY -m unittest discover -s tests -p test_offline_handoff_contract.py -v
$PY -m offline_handoff fixture /path/to/new/basic-bundle --group basic
$PY -m offline_handoff validate /path/to/new/basic-bundle
```

Choose a new output directory whose parent already exists. Creation never
replaces an existing directory. Validation reads the exact fixed artifact set
and refuses symlink artifacts and extra files. Failure during file writing can
leave an incomplete directory; preserve it for inspection, do not treat it as a
valid bundle. Do not run the aggregate suite for this task: existing core tests
execute fitting/prediction. No new dependency, model or dataset is needed.

`synthetic_bundle(group='basic'|'full', zero_targets=False)` returns manifest
bytes plus a dictionary of exact artifact bytes. `validate_bundle(...)` accepts
these bytes and returns a boundary-only report. `adapt_candidates(...)` takes
contexts, raw candidate records, explicit feature config and candidate policy;
it returns core-shaped records plus sidecar provenance. These are proposed APIs,
not a replacement for pending team interfaces. There is no fit/predict command.

## Explicit schema choices

`manifest.json` has exactly the fields enforced in `validate_manifest`: schema
and contract IDs, status, dataset ID, split role/hash, generator/feature-code
commits, feature spec, checkpoint hash, tokenizer `{id, revision}`, candidate
policy, core commit, ordered feature registry and `artifact_sha256` map.
The map contains every artifact below, excluding the manifest itself. SHA-256
checks exact saved UTF-8 bytes, including whitespace; it does not authenticate
who generated them. JSON duplicate keys, nonfinite numbers, exponent overflow,
unpaired Unicode surrogates and undeclared fields are rejected.

| Artifact | Exact proposed shape / isolation |
| --- | --- |
| `contexts.json` | Array of `{context_id, article_id, text}` |
| `candidates.json` | Array of `{question_id, question, article_id, baseline, occurrences}` |
| `questions.json` | Array of `{question_id, candidates}`; each candidate exactly `{candidate_id, answer, features}` |
| `provenance.json` | Generated per-question `{question_id, baseline_candidate_id, origins, selected}` |
| `targets.json` | Separate `{question_id: {candidate_id: number}}`, exact retained-candidate coverage |
| `label-audit.json` | Status, gold references, scorer provenance, target units/origin, multi-gold rule, no-signal policy |
| `split-manifest.json` | Schema ID, content scope, complete article evidence, partition memberships |
| `mapping-illustration.json` | `{status: "illustration_not_prediction_result", mapping: {question_id: answer}}` |

The mapping artifact is a **baseline-format illustration, not a prediction
result**. It deliberately is not named `predictions.json`. It checks exact
question coverage and supplied answer text only; no scorer is invoked.

Each baseline has exactly `{context_id, start, end, answer}`. Each occurrence has
exactly those four fields plus `window_index`, `start_token_index`,
`end_token_index`, `is_context`, `start_logit`, `end_logit`, `combined_score`.
`is_context` is a structural JSON boolean; feature/target/index numbers must not
be booleans. IDs are nonempty Unicode scalar strings, without colons, C0 control
characters or leading/trailing whitespace. Colons are reserved for generated
candidate IDs `context_id:start:end`.

## Deterministic adapter rules

Offsets are zero-based Unicode code-point half-open slices of exact context
text. No normalization or whitespace rewriting occurs. Supplementary characters
count once; UTF-16 or byte offsets must be explicitly converted upstream.
Structurally malformed records, bad score sums, missing contexts, conflicting
window/token origins and valid-bounds answer/slice mismatches fail the batch.
Structurally valid non-context, reversed/out-of-range or invalid/overlength token
spans are filtered before deduplication. Reasons and every original occurrence
remain in provenance, in source input order (including duplicates and spans
excluded by top 20). The selected-winner list is independently recomputed.

Deduplication uses context/start/end, never answer text. Winner order is maximum
combined score, then minimum window/start-token/end-token indices. Conflicting
records with identical context/window/token coordinates are rejected, avoiding
an unresolved final tie. The complete winning score triplet is retained.
Combined score must equal the parsed Python numeric `start_logit + end_logit`
exactly, and the sum must be finite; no tolerance, softmax or renormalization is
introduced. This explicit draft choice may need reconciliation upstream.

The validated baseline span is slot 0 using its dedup winner, even when its score
is lower. Other unique spans sort by score descending, then start/end/context ID
ascending (Python Unicode code-point string order); retain at most 19 more.
Zero valid spans raise `ContractError` with a `candidate_generation_failure`
receipt and origins; no partial successful batch is returned. Missing/filtered
baseline or mismatched baseline text fails rather than inventing a candidate.
Nonempty valid slices exclude empty answers, even though the numerical core has
a broader text interface.

Feature order is fixed in the registry, not inferred from JSON object order:

```text
basic: bert_start_logit, bert_end_logit, span_token_count,
       answer_has_digit, answer_initial_upper
compatibility: q_who, q_what, q_when, q_where, q_why, q_how,
               q_which, q_other, when_digit_match, who_initial_upper_match
```

Basic contains only the first five keys; full contains all 15. Both ordered lists
and the exact spec ID stay in the manifest for either group. Equal-width spec or
order changes fail. Derived features are recomputed from raw supplied evidence;
unknown/missing keys and changed values fail. Finite numeric `1` and `1.0` are
equivalent, but JSON `true` is never a numeric feature or target.

Token count is inclusive `end_token_index - start_token_index + 1`; synthetic
positions are not verified BERT tokenizer positions. ASCII digits alone set the
digit flag; the first ASCII alphabetic answer character determines uppercase.
Question typing is anchored `[A-Za-z]+` after `str.lstrip()`, lowercased, with a
fixed eight-category vocabulary. `WHO` maps to who; `whoé` has initial ASCII
run who; `?Who`, `Whoever` and `In which` map to other. Interactions multiply the
corresponding numeric flags. These lexical rules do not recognize persons/dates.

## Provenance and label limits

Synthetic status is `synthetic_illustration_not_production_valid`. Unavailable
model/generator/feature-code/tokenizer/scorer identities use the explicit
`unavailable:synthetic-no-model-or-scorer` sentinel, never invented hashes.
Artifact/content hashes and the pinned core commit are real. The fixture's
1.0/0.0 targets are **hand-authored illustrations, not scorer outputs**. Its gold
reference belongs only to label-audit. The audit cannot change adapter features.
Allowed feature names and recomputation cannot authenticate upstream logits or
prove that an upstream producer never used gold-derived information.

Unknown production provenance fails closed. This draft requires all listed
artifacts, including targets/audit, and refuses **all production labeled bundles**
until the exact supplied scorer, source hash, invocation and multi-gold rules
are resolved and reviewed. It is not a general production acceptance gateway.
The unavailable scorer was not retrieved, replaced or executed. No token-F1
implementation or numeric conformance expectations are supplied.

The split manifest declares `complete_exact_article_text`, with articles
`{article_id, text, sha256}` and partitions `{role, question_ids, article_ids}`.
Synthetic bundles have only the synthetic partition. Production split checking
requires bert_train, reranker_train, internal_validation and official_dev.
Question IDs and article IDs must be disjoint; identical exact article text
under different IDs across partitions also fails. Contexts must be contained in
assigned articles; identical context text under distinct articles is flagged by
rejection for provenance review. These checks rely on supplied completeness
claims, do not detect semantic/near duplicates, and do not establish real-corpus
disjointness. Article hashes use exact UTF-8 text, without normalization.

`validate_role` checks declared fit/selection/final consumers without executing
them. Only reranker_train can reach the fit boundary; only internal_validation
can reach the selection boundary; official_dev is final-only. Targets have exact
post-truncation question/candidate coverage and numeric values in [0,1]. Single,
equal-zero, equal-nonzero and varying groups are reported without changing
inputs. Core scaling, ranking-loss participation, no-signal default failure and
explicit diagnostic fallback remain deferred; metadata says
`retain_inputs;model_behavior_deferred`. No new fallback is implemented.

## A1–A11 coverage boundary

These classifications describe this offline implementation. Test receipts belong
in the task evidence report; they are not claims of model or CI validation.

| ID | Status | Covered here / deferred |
| --- | --- | --- |
| A1 Identity | Covered | Exact ID coverage, duplicates/missing/extra IDs rejected |
| A2 Ordering/ties | Partial | Adapter dedup/sort/baseline order; reranker score ties deferred |
| A3 Shortlists | Covered | Filters, exact spans, dedup, full winner triplet, top 20, failure receipts |
| A4 Text | Partial | Exact slices/UTF-8/illustration mapping; model prediction serialization deferred; empty spans rejected |
| A5 Separation | Partial | Allowlist/recomputation, separate labels/audit; authentic upstream no-gold provenance unverified |
| A6 Basic/full | Covered | Exact 5/15 ordered vectors and exclusion of all question-type fields from basic |
| A7 Transforms | Deferred | Role gate and fixed taxonomy checked; fitting/scaler and inference immutability unexecuted |
| A8 No signal | Partial | Label classification and unchanged inputs; core scaling/loss/fallback behavior deferred |
| A9 Evaluator | Deferred | Real-label refusal checked; supplied-scorer conformance blocked/unexecuted |
| A10 Splits | Partial | Declared question/article/exact-content overlap and role checks; real-corpus evidence unverified |
| A11 Round trip | Partial | Artifact hashes/spec/order and illustrative mapping serialization; model reload scores/predictions deferred |

Next: review the provisional schema/feature/offset/score choices with interface
owners and identify the supplied scorer through an authorized source before any
real-label integration. No such coordination, training, evaluation or publication
is performed by this task. Source and tests were generated with Codex assistance
and independently reviewed by another Codex agent.
