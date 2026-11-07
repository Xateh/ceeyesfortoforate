# Reranker guide

The CPU reranker selects one supplied BERT candidate per question. Use
[`reranker.real_training`](../reranker/real_training.py) for real labels and
[`team_candidates`](../reranker/team_candidates.py) for Sean's schema-1 exports.
The baseline is the fine-tuned BERT span selector, not another learned reranker.
Its candidates use raw start/end logits across valid windows without max-context filtering.

## Inputs

Use the team's seed-4248, 75/15/10 article split. Supply these seven JSON files
as `inputs` entries in the run manifest, each with `path` and exact `sha256`:

| Key | File |
| --- | --- |
| `source_train` | Original training `train-v1.1.json` |
| `bert_train` | `data/splits/bert_train.json` |
| `reranker_train` | `data/splits/reranker_train.json` |
| `internal_val` | `data/splits/internal_val.json` |
| `train_candidates` | `reranker_train_candidates.json` |
| `validation_candidates` | `internal_val_candidates.json` |
| `validation_baseline` | Matching `internal_val_predictions.json` |

Team artifacts are in the repository's [releases](https://github.com/Xateh/ceeyesfortoforate/releases):
`data-v1`, `bert-finetuned-v1` and `candidate-exports-v1`. The adapter targets
exporter revision `e56efaa1aacfa5d547286137375ab398baaeefeb`. No separate training
baseline file is required; training first answers are derived from the candidates.
Fitting from exports needs no BERT execution or model-weight loading.

Before fitting, the runner checks the full source/split union, exact identities,
disjoint question/article/exact-context sets, complete candidate coverage and
validation baseline parity. It preserves supplied order and declared float32/float64
logit sums and 1–20 unique spans per question. Spans are exact Unicode code-point
slices `[start, end)`; identical text at different offsets remains distinct.
Only source-to-export question comparison
applies the producer's `.strip()`; source/split text, contexts, answers and IDs stay exact.

## Features and fitting

Feature spec `candidate-features/0.1.0-draft` has fixed order:

- **Basic (5):** `bert_start_logit`, `bert_end_logit`, `span_token_count`,
  `answer_has_digit`, `answer_initial_upper`.
- **Full (15):** basic, then `q_who`, `q_what`, `q_when`, `q_where`, `q_why`,
  `q_how`, `q_which`, `q_other`, `when_digit_match`, `who_initial_upper_match`.

Digit and uppercase checks use ASCII; uppercase refers to the first ASCII letter.
Question type is the first ASCII alphabetic token after leading whitespace, otherwise
`other`. Interactions multiply `q_when` by the digit flag and `q_who` by the uppercase
flag. Basic excludes all question-type features. The eight raw type indicators are
constant within each question and cancel in this centered linear objective.

Targets are separate candidate F1 values in [0,1], generated only from training golds
by the pinned repository scorer. It takes the maximum over nonempty normalized golds,
using an empty-string fallback if none remain. Golds and targets never become features.
The core fits training-only means and population standard deviations (constant scale 1),
then float64 ridge on within-question-centered features and targets, equally weighting
informative questions. Scores are ranking values, not probabilities; exact ties choose
first candidate. Singleton/all-equal groups remain in scaling and coverage but not
ranking loss. A wholly uninformative batch fails; the real runner has no silent fallback.

## Manifest and execution

Copy [`configs/reranker-real.example.json`](../configs/reranker-real.example.json)
to an untracked run manifest and fill its unresolved values. Paths resolve relative
to that manifest. The unchanged template intentionally fails validation.
Top-level fields are `schema`, `status`, `policy`, `inputs`, `scorer`, `lineage`
and `baseline_generation`; real runs use status `real_provisional`.

`policy` fixes the ordered registry, target, `no_signal="error"`, and 1–10 distinct
positive `alphas`. Keep its required `id`, `decision_reference`, `acceptance` and
`selection` fields; they describe the run configuration, not team agreement.
For each variant, the runner selects highest internal-val F1, then EM, then first
configured alpha. Use a predeclared grid.
The checked run used `[1.0]`. Freeze settings before any official-dev comparison.
`scorer.hashes` pins both `evaluate-v2.0.py` and `scripts/squad_eval.py`; the
repository implementation is available, but course-source equivalence is unverified.

**Strict provenance is the default.** Fill the template's hashed checkpoint/tokenizer
file descriptors and `run_info`, `train_log`, `environment`, relationship reference
and matching checkpoint/generation declarations. Hash checks do not load weights.

**Explicit provisional mode** requires both the flag below and a different lineage
object: schema `cs4248.provisional-provenance/0.1.0-draft`, status
`provisional_team_declarations_unverified`, `approval_reference`, `evidence_references`,
`checkpoint_export_value`, hashed `archives` for data/checkpoint/candidates,
`original_records` marking run_info/train_log/environment `unavailable`, and `unverified`
marking training lineage/environment, base-model/tokenizer revisions and independent
baseline execution `unverified`. Replace `baseline_generation` with status
`supplied_predictions_execution_unverified`, `declared_settings`, `unverified_fields`
and `evidence_references`. Unknown fields must be the ordered complement of known
settings in `BASELINE_FIELDS`; declared settings must match the export exactly.
See the [schema checks](../reranker/real_training.py) and
[fixture construction](../tests/test_real_training.py) for exact keys. Do not invent logs.

Run inside an appropriately sized CPU allocation from the repository root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python -m reranker.real_training --manifest run-manifest.json --output outputs/reranker
# For an explicit provisional manifest, add: --allow-provisional-provenance
```

The output directory must be new, with an existing parent. Inputs are not rewritten.
The CLI does not allocate compute or enforce resource limits; decoded exports can
occupy several GiB. The recorded alpha-1 run peaked at about 3.1 GiB resident memory.

## Outputs and reload

A successful run writes `complete.json`: baseline/basic/full EM/F1, trial results,
selected alphas, hashes and provenance. Other outputs are `manifest.json`, separate
`targets.json`, derived training first answers, baseline predictions and each variant's
`*-model.json`, `*-binding.json`, `*-predictions.json`. Missing `complete.json` means
an incomplete run; keep partial outputs for diagnosis and use a new output directory.

Reload takes label-free candidate bytes and an independently retained binding hash:

```python
from pathlib import Path
from reranker.real_training import reload_predict

answers, scores = reload_predict(
    Path("outputs/reranker"), "basic", trusted_binding_sha256,
    Path("new_internal_val_candidates.json").read_bytes(),
    allow_provisional_provenance=True,  # only for an explicit provisional binding
)
```

`answers` maps every question ID to its exact supplied text. Reload never fits or
updates preprocessing. It checks model/source hashes, feature identity/order, generation
settings, internal_val role and BERT/reranker-training exclusions. New compatible
batches need not match old validation bytes. A hash inside its own artifact is not
an independent trust anchor. The real runner has no official-dev inference path.

## Tests and current result

Run each suite in a fresh process; the offline contract tests check that model modules
were not imported. These commands use fixtures, not release downloads or BERT inference:

```bash
for suite in test_reranker.py test_team_candidates.py test_team_model.py \
  test_real_training.py test_offline_handoff_contract.py test_handoff_integration.py \
  test_heldout_inference.py test_evaluator_boundary.py; do
  python -m unittest discover -s tests -p "$suite" -v || exit 1
done
```

The published provisional alpha-1 run passed 18 focused real-training tests and
exact model reload checks. On all 9,131 internal-validation questions:

| System | EM (%) | F1 (%) |
| --- | ---: | ---: |
| Supplied BERT baseline | 65.403570 | 79.287794 |
| Basic, alpha 1 | 63.925090 | 76.781898 |
| Full, alpha 1 | 63.892235 | 76.772235 |

Neither variant beat baseline. Basic corrected 339 EM errors but broke 474; full
corrected 337 but broke 475. These are provisional results, not an accuracy gain.
Original BERT training/baseline execution, exact upstream revisions/environment and
course-scorer identity remain unverified. Internal validation was used upstream for
BERT validation, so it is not untouched. Hashes establish byte identity, not origin;
exact-context checks do not detect paraphrases. Features and target policies remain
provisional rather than team-agreed. No reranker official-dev result is claimed.

## Other APIs and contribution

`Reranker` is the generic numerical core; it does not enforce split provenance.
`TeamCandidateModel.fit_synthetic` remains synthetic-only. `SyntheticRun` replays the
older draft bundles; `HeldoutRun` accepts new label-free synthetic bundles. These
are compatibility/test APIs, not substitutes for the real-label runner above.
See their source in [`reranker/`](../reranker/) and [`offline_handoff/`](../offline_handoff/).

Xu An's reranker contribution covers the stable ridge core, candidate validation,
basic/full feature separation, real-label integration, persistence and validation,
building on Hai's pipeline/evaluator and Sean's exporter. Codex generated and edited
implementation, tests and text, and assisted technical review. Xu An directed the
project work; report contributions and AI use should be reviewed and attributed accurately.
