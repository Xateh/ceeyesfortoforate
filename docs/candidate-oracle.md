# Candidate oracle diagnostic

Measure answer quality available in existing top-20 candidates before tuning
the reranker. Gold answers are used for analysis only; there is no training,
BERT inference or official-dev evaluation.

From the repository root, with the project's dependencies installed:

```bash
mkdir -p outputs
python -m scripts.candidate_oracle \
  --dataset data/splits/internal_val.json \
  --candidates outputs/inference/internal_val_candidates.json \
  --baseline outputs/inference/internal_val_predictions.json \
  --output outputs/internal_val_oracle.json
```

Adjust paths to the extracted files. The output must be new and its parent
must exist. Inputs are not modified. The script reuses the schema-1 adapter,
dataset validator and repository scorer. It checks full coverage, matching
article/context/question identities (exported questions use `.strip()`), valid
ordered spans, declared `internal_val` role and baseline/first-candidate parity.
These checks do not independently verify true split role or training lineage.

## Report fields

| Field under `summary` | Meaning |
| --- | --- |
| `baseline.exact`, `baseline.f1` | Baseline EM/F1, in percent |
| `oracle.exact` | Percent with any normalized exact-match candidate |
| `oracle.f1` | Mean best candidate token F1 per question, in percent |
| `headroom_percentage_points` | Oracle minus baseline |
| `baseline_em_errors` | Questions baseline did not answer exactly |
| `recoverable_em_errors` | Baseline EM errors with an exact candidate available |
| `unrecoverable_em_errors` | Baseline EM errors without an exact candidate |
| `recoverable_fraction_of_em_errors` | Recoverable/errors in [0,1]; null if no errors |
| `candidate_count_histogram` | Questions at each actual candidate count |

`oracle_by_k` gives cumulative oracle scores at every available rank, from 1
to the largest list (at most 20). Short lists use all available candidates and
every question stays in the denominator. `per_question` gives baseline/oracle
scores in [0,1], candidate count and first exact-match rank (null if absent).
EM and F1 maxima are independent upper bounds; no oracle predictions are
emitted. Normalization, multiple-gold scoring and empty-normalized-gold fallback
come from the repository scorer. Input and software SHA-256 hashes are recorded;
course-scorer equivalence and checkpoint/tokenizer provenance remain unverified.

A large oracle gap indicates room for better ranking, though existing features
may not learn it. A small gap limits the benefit of ranking these candidates.
Use this evidence to predeclare the next experiment and freeze configurations
before the official-dev comparison.

## Validation and real-data status

```bash
python -m unittest discover -s tests -p test_candidate_oracle.py -v
```

Tests use invented examples with hand-checked metrics, short lists, multiple
golds, normalization, input mismatches and CLI overwrite protection.
All 42 tests passed across the new diagnostic, team candidate adapter,
real-training and evaluator-boundary suites (7/12/18/5, separate processes).
The real archive also passed adapter validation and first-candidate/baseline
parity on all 9,131 internal-validation questions, each with 20 candidates.

No real oracle result is recorded yet. The available `inference.zip` has
candidates and baseline predictions, but no gold dataset. Its SHA-256,
`5505d15d37b26269a383c8adb6ce22ffecced629b636312ebd77cd3ffd19baf1`,
matches the `candidate-exports-v1` release. The remaining input is the team's
`data/splits/internal_val.json` from `data-v1`. Run the command with that file
before drawing a numerical conclusion about ranking headroom.
