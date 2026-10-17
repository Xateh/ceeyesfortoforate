# Reranker core — provisional issue #10 interface

This is a bounded, synthetic-data implementation, not completion of the actual
training/evaluation work in issue #10. It is based on setup PR #17 at
`c97807538f6ab4dea32c350a65c96b8ae20cae4e`. The user approved this local
dependent branch before setup merged; no assignment, push, or PR is part of this
work. Requirements from issues [#10](https://github.com/Xateh/ceeyesfortoforate/issues/10)
and [#9](https://github.com/Xateh/ceeyesfortoforate/issues/9) were read on
2026-10-17. The [final proposal](https://docs.google.com/document/d/1OJk-LjXQUl51oHKlchZh3eJXMWZLE0E3yYZVBI4THFo/edit)
describes overlap supervision and comparison with/without question-type features.
Neither source fixes a shared file schema or exact objective.

## Adapter boundary

Import `Candidate`, `Question`, `FeatureConfig`, and `Reranker` from
`reranker`. A candidate has a nonempty candidate ID, supplied answer text,
and a mapping of named finite numeric features. A question has a nonempty
question ID and a nonempty ordered candidate sequence. IDs are unique within
their respective question/batch. A single candidate is valid. Answer strings
are preserved exactly, including Unicode and empty strings. Candidate generation,
span validity, feature extraction and question-type encoding remain upstream.

`FeatureConfig(basic=("score", "length"), compatibility=("type_match",),
group="basic" or "full")` explicitly lists ordered feature names. These example
names are placeholders, not a team agreement. Basic uses only the basic list;
full adds the compatibility list. The adapter must classify **all** question-type
features (including raw question-type encodings) as compatibility if the intended
ablation removes all question-type information. No features are discovered from
data. Extra feature keys are ignored, including invalid values; missing selected
features and invalid selected values fail. Categorical features must be encoded
upstream with a fixed vocabulary or training-only fitting.

Targets are a **separate** mapping:
`{question_id: {candidate_id: score}}`, with exact ID coverage and finite scores
in [0, 1]. A provisional interpretation is candidate token-overlap F1 against
the gold answers, produced by #9. This code does not compute labels, read gold
answers, split data, or implement an evaluator. Token normalization, multiple
gold-answer aggregation and shortlist/no-useful-answer policy need agreement
with #9 and the evaluation owner.

Known gold/label/target/F1/EM feature-name tokens are rejected in configuration.
This is a guardrail, not proof of provenance: the adapter must never disguise
gold-derived information under an allowed name. Answer text and IDs are never
converted to features. Only the explicit feature allowlist enters the scorer.

## Provisional objective and edge cases

The architecture is a linear CPU float64 scorer, using PyTorch already pinned
by PR #17. No new runtime dependency is introduced. Training fits per-feature
mean and population standard deviation using **all supplied training candidates
only**; constant features receive scale 1. Validation is not accepted by
`fit` and inference never updates preprocessing.

Let z be standardized features. Within each informative training question,
subtract that question's mean feature vector and mean target. Minimize:

`mean_over_informative_questions(mean_over_candidates((centered_z @ w -
centered_target)^2)) + alpha * ||w||^2`

The default positive ridge penalty is `alpha=1`; it is configurable. This
centered least-squares objective focuses on relative candidate quality and gives
each informative question equal total loss weight despite variable shortlist
lengths. It is a provisional choice, not an agreed team objective or a claim
of validation-selected hyperparameters. Inference scores are uncalibrated ranking
values, not probabilities. There is no intercept because it cannot change rank.

Single-candidate questions and questions whose targets are all equal (zero or
nonzero) do not contribute ranking loss. They still contribute to train-only
scaling. If **all** training questions lack target variation, fit fails by
default. Explicit `no_signal="first"` creates a zero-weight fallback and records
zero informative questions. If features cannot distinguish candidates despite
varying targets, the ridge solution may also tie. All exact score ties always
select the first candidate in the supplied order. There is no gold-based
fallback, answer synthesis, or abstention. The adapter must provide a stable
order (e.g. its existing baseline order) to make ties reproducible.

`select` returns one of the supplied Candidate objects for each question;
`predict` returns exactly `{question_id: answer_text}`, the intended supplied
SQuAD-scorer prediction mapping. Empty question batches, zero-candidate questions,
duplicate IDs, malformed records and incomplete labels fail explicitly rather
than silently dropping questions. `save_predictions` writes UTF-8 JSON.

## Persistence and local CLI

`save` / `Reranker.load` round-trip weights, preprocessing mean/scale, ordered
feature groups, alpha, no-signal policy, objective/version and training counts in
a single plain JSON file. Loading validates fields, dimensions, finite numbers,
positive scales and counts. It does not deserialize executable pickle objects.
Use the same feature semantics at prediction time; a JSON model cannot verify
upstream feature provenance or units.

The optional CLI's JSON records are only an example **local adapter format**.
Replace `questions_from_records` at the integration boundary when #9 settles its
interface. It intentionally rejects extra question/candidate record fields, so
target annotations cannot silently enter via those records.

Example questions file:

```json
[
  {"question_id": "demo", "candidates": [
    {"candidate_id": "a", "answer": "first", "features": {"score": 0, "type_match": 0}},
    {"candidate_id": "b", "answer": "second", "features": {"score": 1, "type_match": 1}}
  ]}
]
```

A separate targets file contains `{"demo": {"a": 0, "b": 1}}`. A features
file contains `{"basic": ["score"], "compatibility": ["type_match"], "group": "basic"}`;
set group to `"full"` for the full variant. These are synthetic examples.

From the repository root in the pinned environment:

```bash
python -m reranker train --questions train.json --targets targets.json \
  --features features.json --model checkpoints/reranker-basic.json
python -m reranker predict --questions validation-inputs.json \
  --model checkpoints/reranker-basic.json --output outputs/predictions-basic.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Train only on the separate reranker-training partition. Generate validation
predictions without passing its labels to fit; use the supplied scorer maintained
by the evaluation owner. Choose any hyperparameters on internal validation and
keep official dev-v1.1 for the final evaluation. This module cannot detect
upstream article overlap or enforce split provenance; #2/#9 must supply that
contract. Actual SQuAD/BERT training, candidate generation, feature engineering,
split creation, ablation evaluation and final scoring have not been run here.

## Review and next integration step

Synthetic tests verify both variants, variable counts, compatibility exclusion,
constant features, no validation-statistic mutation, deterministic ties,
save/reload equivalence, malformed/missing inputs, no-signal fallback and
separation of labels from inference inputs. See `reranker-validation.md` for
the actual environment, commands, independent review and outcomes.

Before real training, agree with #9 on target definition, all-tied/no-useful
candidate policy, the adapter and feature-group lists; then connect a small
real labelled fixture from the approved reranker-training split and its separate
internal-validation inputs. Coordinate the output contract with the evaluator
owner. Do not claim an accuracy gain from these synthetic software checks.

This implementation and tests were generated with OpenAI Codex assistance and
independently reviewed by a second Codex agent. No external source code was
copied; PyTorch supplies tensor operations and the ridge linear solver.
