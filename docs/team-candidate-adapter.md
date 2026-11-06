# PR28 candidate-to-reranker adapter (provisional)

`reranker/team_candidates.py` consumes schema_version 1 from PR28 head
`e56efaa1aacfa5d547286137375ab398baaeefeb`. Its source, exporter and docs were
inspected on 5 November 2026; PR28 was open/unmerged. Main was
`d11a5ea32cc1f213001b378d048c2c0550e7130b`. Reported real export counts and
baseline agreement are teammate reports, not validation performed here.

This additive adapter preserves supplied order, including equal-score candidates
and repeated answer strings at different offsets. It never sorts, rededuplicates,
regenerates spans or applies the older draft's tie rules. Context IDs use the
team's exact JSON title/context hash (including default JSON separators).
Strict schemas reject extra fields, including injected features/golds/targets.
It checks exact Unicode slices, 1-20 unique spans, integer token counts/indices,
global window ownership, descending exported scores and declared float32/float64
addition. Exported float32 sums are not replaced with Python float64 sums.
Unsupported dtypes/policies fail. No upstream winner or discarded origin is invented.

## Label-free use

```sh
python -m reranker.team_candidates --input internal_val_candidates.json \
  --output adapted-full.json --group full --baseline internal_val_predictions.json
```

The baseline argument is optional. When supplied, its IDs must match exactly and
every first answer must match. This checks answer text, not span identity for
repeated strings. Without it, baseline agreement remains explicitly unverified.
The output file must not exist; existing files, including inputs, are preserved.
No model, tokenizer or scorer is called by conversion.

Output contains `profile_id`, `feature_registry`, strict core `questions`, and
separate `audit` metadata. Pass only `questions` to `questions_from_records`.
The audit retains the exact export SHA-256, producer revision inspected (not an
attestation of which producer ran), supplied metadata and retained occurrences.
Paths are declarations, not checkpoint/tokenizer/split content hashes. This is
not an old PR20 validated bundle and does not use its candidate policy validator.

## Provisional features and synthetic execution

The existing `candidate-features/0.1.0-draft` definitions are reused unchanged:
basic has start logit, end logit, token count, ASCII digit flag and initial ASCII
uppercase flag. Full appends the fixed eight question-type indicators and two
interactions. All question-type fields are excluded from basic; supplied feature
keys are never accepted. These definitions remain a proposal pending #8/#9.

`prepare_pair(train_bytes, validation_bytes, group=...)` enforces reranker_train
and internal_val roles, matching declared generation settings, and disjoint
question IDs, exact article titles and exact context hashes. It does not establish
source authenticity, near-duplicate safety or membership in the merged split.
Official dev is refused by this adapter; dev has already been reported evaluated
by the team and is not described as untouched.

`fit_synthetic_pair(train_bytes, targets, validation_bytes, group=...,
target_status='synthetic_hand_authored_not_scorer_outputs')` calls that guard
before the existing core fit. Targets remain a separate exact-ID mapping, with
no-signal default failure and no implicit fallback. The status is an explicit
caller attestation, not a detector of real versus synthetic data. This entry point
is for tiny invented fixtures only, not an approved real-label training path.

`fit_synthetic_pair` returns the standalone core, whose persistence does not bind
this adapter profile or upstream provenance. For adapter-aware synthetic
save/reload, use the implemented [TeamCandidateModel wrapper](team-model.md).
It binds the exact feature registry, declared generation identity and training
membership; it still does not authenticate checkpoint/tokenizer bytes or enable
real-label training. Existing SyntheticRun/HeldoutRun retain their older draft
bundle formats and are not interchangeable with PR28 exports. See the
[current reconciliation](reranker-status.md) for the active paths and dependencies.

The focused synthetic suite is `python -m unittest discover -s tests -p
'test_team_candidates.py' -v`. It exercises conversion plus actual tiny core
fit/predict/save/reload, first-supplied ties, Unicode prediction bytes, variable
candidate counts/no-signal groups, train-only transforms, bad coverage/roles,
dtype-sensitive score relationships, and the basic/full CLI. It does not test
real BERT accuracy, reconstruct tokenized windows or execute any evaluator.

Next inputs: PR28 export bytes and baseline answer mappings, verified source
split/checkpoint/tokenizer identities, and accepted feature/target definitions
with supplied-scorer provenance. The merged seed-4248 75/15/10 split governs
assignments; the old seed-42 pilot is retained only as historical evidence.
Team BERT/evaluator implementations are not duplicated. No real-label fit or
official-dev execution was performed for this increment. The adapter was subsequently
published; see the dated reconciliation for current branch status.
