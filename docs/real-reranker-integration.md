# Real-label integration — provisional run policy

`python -m reranker.real_training` adds a separate real-label workflow. It does
not weaken `TeamCandidateModel.fit_synthetic`, generate BERT candidates, download
assets or use official dev. Release candidate/baseline inputs are now available.
Strict mode retains its original run-record requirements. An explicitly opted-in
provisional mode can use validated exports with recorded missing provenance,
with explicitly recorded provenance limitations.

## Required handoff

Use the team's Canvas/seed-4248 75/15/10 article assignments, not the historical
HF/seed-42 split. BERT optimizer seed 42 is a different setting. Supply exact bytes
and SHA-256 for all seven inputs, with paths relative to the manifest:

| Manifest key | Expected source |
| --- | --- |
| `source_train` | Original training-only SQuAD `train-v1.1.json` from the verified release |
| `bert_train` | `data/splits/bert_train.json` |
| `reranker_train` | `data/splits/reranker_train.json` |
| `internal_val` | `data/splits/internal_val.json` |
| `train_candidates` | Sean's `outputs/bert-base-baseline/reranker_train_candidates.json`, schema 1 |
| `validation_candidates` | Matching `outputs/bert-base-baseline/internal_val_candidates.json`, schema 1 |
| `validation_baseline` | Supplied `outputs/bert-base-baseline/internal_val_predictions.json` (separate execution is producer-declared) |

No independent training-split baseline is required: that baseline mode only
supports internal_val/dev. Training first-candidate answers are saved as derived,
not independently verified baseline outputs. Validation IDs and first answers
must match the supplied baseline exactly. Strict mode requires all generation settings to be recorded and equal the
candidate settings. Provisional mode records only supported declarations, checks
those exactly, and lists all remaining fields as unverified. Neither declarations
nor answer parity authenticate the original baseline execution.

Full supplied split/source identities are checked before fitting: exact question,
article, context and gold-answer identity; complete union of training-source IDs;
unique IDs/titles; disjoint question/article/exact-context sets across all three
roles; exact gold and candidate Unicode spans; complete candidate coverage for
reranker_train and internal_val. No sampling is used for these gates. This proves
consistency of the supplied package, not that its bytes came from Canvas or that
near-duplicate contexts are absent. The runner consumes assignments; it does not
reshuffle or reimplement the team's split allocation.

Source-to-split question text remains exactly equal as a Unicode string. At the
candidate boundary only,
PR28 exports the question produced by `scripts/preprocessing.py` using
`qa["question"].strip()`. Require the exported question to equal exactly that
transformation of the source question; do not trim the export, normalize internal
whitespace, or normalize article/context/answer text. Context offsets, golds, IDs
and candidate order retain their exact checks. This compatibility correction is
bound through the saved runner source hash; input files are never rewritten.

Strict real mode additionally requires hashed checkpoint and tokenizer file descriptors,
run_info.json, train_log.jsonl and environment.txt, plus an explicit relationship
reference and checkpoint declaration matching both exports. Files are hashed
without loading model weights. These checks identify supplied bytes; semantic
checkpoint/tokenizer/export correspondence remains a caller-attested relationship
that must be reviewed against the actual release and run records before a real run.
The verified release includes tokenizer files; the archive inspection below records
what is established and what lineage remains unresolved.

## Explicit policy and scorer

Copy `configs/reranker-real.example.json` outside the tracked template and fill
all unresolved nulls from verified inputs and a recorded run decision. The
unchanged template fails closed. A decision reference identifies the selected
experiment policy; it is not a claim of team agreement.

The supported registry is the existing provisional basic-five/full-fifteen spec:
basic excludes every question-type encoding and interaction. Unknown/equal-width
changed specifications fail. Configure an ordered positive ridge-alpha grid
(up to ten entries). Target policy is explicitly token F1 from the pinned team
scorer, with its own nonempty-normalized multi-gold maximum and empty-gold fallback.
Target values are in [0,1]; reported EM/F1 are percentages. Golds and targets never
enter feature or inference records. No-signal fallback is forbidden: single and
all-equal groups retain coverage/scaling but supply no ranking loss; a wholly
uninformative batch fails.

The runner delegates to `scripts/squad_eval.py` and `evaluate-v2.0.py`, checking
both source hashes first. It does not reconstruct or substitute the evaluator.
The repository source is available; equivalence to the exact course download is
still unverified and is explicitly recorded. Fixture checks execute these repo
bytes; equivalence to the original course download is not established.

Main's actual `scripts/bert_predict.py` has no max-context filtering: all valid
windows compete by raw logits. Preserve the PR28 traversal order and declared
float32/float64 sum, despite the older main documentation's contrary statement.

## Execution, selection and persistence

After artifact/lineage review and resource sizing, use the
existing environment from the repository root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python -m reranker.real_training --manifest /absolute/path/run-manifest.json \
  --output /absolute/path/new-run-directory
```

The output directory must be new, with an existing parent. The runner never
rewrites inputs or an existing run. A failed fit may leave diagnostic partial
outputs; absence of `complete.json` means the run did not complete. The template
does not fetch data or allocate compute resources.
The caller must apply appropriate process memory/runtime/CPU limits; the CLI
does not allocate resources or raise limits itself.

Targets are generated only from reranker-training golds through the team scorer.
For each variant/alpha, the unchanged centered ridge core fits and scales using
training candidates only. Validation predictions precede validation scoring.
Choose highest internal-val F1, then EM, then first configured alpha for exact
ties. Every trial and the selected alpha are recorded. Freeze that selection
before any future official-dev comparison; this runner has no official-dev path.

Outputs include separate targets, derived training first answers, the supplied
validation baseline mapping, basic/full predictions, model JSON and binding JSON,
and a final report with baseline/basic/full metrics and deltas. Each binding
contains policy, ordered feature identity, input/target hashes, generation
metadata, source-code hashes and BERT/reranker-training exclusions. Gold text and
target values stay outside model and inference inputs.

`reload_predict(directory, group, expected_binding_sha256, candidate_bytes)`
requires a separately retained binding hash, checks the model and implementation
hashes, rejects role/provenance/feature mismatches and training overlap, and
accepts genuinely new compatible label-free internal_val batches. It never fits.
Compatibility requires identical bound software bytes, not just schema IDs.
Every selected model is reloaded and checked for exact score/prediction equality;
serialized state must remain unchanged during validation and reload inference.
A hash in `complete.json` is a convenient receipt, not an authenticated external
trust anchor. Retain trusted pins independently. New-batch role and lineage
claims still require authentic upstream provenance.

## Validation and limits

The original `90a2a6fe` increment passed **35 selected fixture/regression tests**:
eight in
`test_real_training.py`, 18 in `test_reranker.py`, and nine in `test_team_model.py`.
These are distinct from the **94 earlier synthetic regressions** recorded for
consolidation commit `619f77fe3cd12f02c2f036da19b94bd476524519`. The unchanged
implementation/test bytes were verified against their passing receipts before
publication; the 94-test suite was not rerun for this increment.

From the repository root, the selected commands were:

```bash
python -m unittest discover -s tests -p test_real_training.py -v
python -m unittest discover -s tests -p test_reranker.py -v
python -m unittest discover -s tests -p test_team_model.py -v
```

The focused suite uses invented nested SQuAD and PR28-shaped inputs. It exercises
the real repository scorer on normalization, multiple golds, partial overlap,
empty-normalized golds and [0,1] targets; complete split/ID/baseline gates;
training-only fits/scaling; selection; all-zero/single groups; no-signal failure;
Unicode, exact ties, new-batch reload, tampering and input preservation. It also
checks that the older synthetic fit guard remains intact. These are software
checks, not real-data training or QA accuracy results.

Full production memory/runtime requirements are not established by tiny fixtures.
Fixture checks used one CPU thread, a 1 GiB process limit and CUDA disabled.
Neither large export materialization nor a full alpha sweep is claimed to fit
that footprint. Size the intended run before execution.

## Verified release inputs, 6–7 November 2026

All three compressed release archives were hash-verified. Seven required JSON
inputs were subsequently materialized; no model weights or official-dev inputs
were extracted:

| Release asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `data-v1/data.zip` | 58,488,042 | `bd9577c8dcd7d192c61d2a802e3b8cff10d71e1b1b57cdc5e64bbc4c9e0d245c` |
| `bert-finetuned-v1/final.zip` | 403,707,864 | `a8cf3d7fe9ec5fc07a240353d7d858cc8e4c0dac76bbbee57c982852e8601662` |
| `candidate-exports-v1/inference.zip` | 21,130,115 | `5505d15d37b26269a383c8adb6ce22ffecced629b636312ebd77cd3ffd19baf1` |

The saved splits contain 332/66/44 articles and 65,952/12,516/9,131 questions.
An archive-only audit compared every article and question with original
`train-v1.1.json` using type-preserving canonical JSON. The split union contains
all 442 articles and 87,599 questions exactly once, with zero article, question,
context, answer or wrapper-metadata mismatches. Duplicate IDs/titles, cross-role
article/question/exact-context overlap and invalid answer spans were not detected.
This establishes consistency with the original training JSON inside the archive;
it does not authenticate Canvas origin, reproduce the seeded allocation, certify
near-duplicate safety or validate candidate exports. Official-dev golds and
serialized tokenized data were not read.

The checkpoint archive contains `final/model.safetensors`, `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`
and `training_args.bin`. Only JSON/text metadata was inspected; weights and
training arguments were not deserialized. Configuration declares BERT QA,
float32, 12 layers, hidden size 768, 12 heads, 512 positions and vocabulary size
30,522. The lowercase WordPiece tokenizer declares the same vocabulary count.
These are compatible declarations, not proof of actual training/export lineage.

The checkpoint archive does not include `run_info.json`, `train_log.jsonl` or
`environment.txt`; the inspected metadata does not establish an exact base
revision. A scoped search of the existing project found only an older CPU
`environment.txt`, without a verified link to the released training run. It is
not a substitute. Original training/export/base-revision correspondence and independent baseline
execution remain unverified. These gaps must stay explicit in any provisional run.

On 7 November, the corrected runner passed **12 focused fixture tests**, including
four question-boundary regressions, in a CPU-only Slurm allocation. Full input
validation then checked all **12,516 training questions / 250,320 candidates** and
**9,131 validation questions / 182,620 candidates**. Every question has 20 valid
candidates. Source/split coverage, question/article/exact-context disjointness,
gold and candidate spans, declared score dtype/sums/order, basic/full feature
membership and values, and matching train/validation generation declarations
passed. All 9,131 validation IDs and first-candidate answers match the separately
supplied baseline file exactly. Matching declarations and answers do not
authenticate how that baseline or checkpoint was produced.

The source has 896 training and 201 validation questions whose outer whitespace
is removed by the verified producer transformation. All exported questions match
that transformation; raw source/split identity remains exact. The inputs were not
rewritten. The successful job used one CPU, an 8-GiB ceiling and 44.364 seconds;
scheduler peak memory was 2,088,992 KiB. This measures input validation and fixture
regressions, not the memory/runtime of real target generation or training.

The provisional experiment uses basic5/full15 features, repository-scorer F1,
`no_signal="error"`, alpha `[1.0]` and internal-validation selection. These policy
choices are not asserted as team agreement. Explicit provisional mode permits
fits while recording missing original run records rather than inventing them.
The synthetic-only API and the default strict real-mode guards remain unchanged.
Completed-run evidence, if any, is recorded separately below.


## Explicit provisional provenance mode

The CLI requires `--allow-provisional-provenance`; the Python `run` and
`reload_predict` APIs require `allow_provisional_provenance=True`. Omitting it
retains the existing strict behavior. A flag alone cannot rescue an incomplete
strict manifest: the explicit provisional schema must also be supplied.

The provisional run manifest uses status `real_provisional` and lineage schema
`cs4248.provisional-provenance/0.1.0-draft`, with status
`provisional_team_declarations_unverified`. It records an explicit experiment-decision reference,
evidence references, the checkpoint path declared in the exports and hashed
archive descriptors for data/checkpoint/candidates. `original_records` marks
`run_info`, `train_log` and `environment` as `unavailable`; `unverified` marks
training lineage, training environment, base-model revision, tokenizer revision
and independent baseline execution as `unverified`. These are statements of
missing evidence, not replacement run records. Archive hashes identify bytes;
they do not authenticate origin or establish cross-artifact lineage.

`baseline_generation` has status `supplied_predictions_execution_unverified`,
`declared_settings`, an exact ordered `unverified_fields` complement and evidence
references. Every declared setting is compared to export metadata with exact
type/value equality. For this run only checkpoint path, max-answer length 30,
n_best 20 and batch size 64 are supported by the team records reviewed. Unknown
baseline settings are not copied from candidate metadata as if independently
verified. Exact supplied baseline ID/answer parity remains mandatory.

Bindings preserve these declarations and uncertainties, and require explicit
provisional opt-in again on reload. Saved checkpoint and baseline declarations
must agree with saved generation identity. Reload uses no labels and never fits
or changes preprocessing. All source/split, article/context/span/ID, feature,
coverage, score, ordering and overlap gates remain active.

Hai's PR26 declares training on bert_train with internal_val used for BERT
validation. Internal-val reranker comparisons therefore are not evidence from an
untouched upstream holdout. Original training logs, exact environment/revisions
and baseline execution remain unreproduced; course-source scorer equivalence is
unverified. No BERT rerun, weight loading or official-dev evaluation is included.


## Provisional internal-validation run, 7 November 2026

Explicit opt-in and the provisional policy above were used for one basic and
one full fit, both alpha 1.0. The expanded focused suite passed 18 tests, including
six provisional-mode tests. The original 94-test consolidation suite was not
rerun. The real CLI completed on one CPU with CUDA disabled in an 8-GiB
allocation. No model weights were loaded.

All 12,516 reranker-training questions and 250,320 candidates were retained.
12,393 questions had varying F1 targets and contributed ranking loss; the other
123 remained in coverage and scaling. Both models saved and reloaded with exact
score/prediction equality and unchanged learned preprocessing. Every output
covered all 9,131 internal-validation IDs.

| System | EM (%) | F1 (%) | EM change (points) | F1 change (points) |
| --- | ---: | ---: | ---: | ---: |
| Supplied baseline | 65.403570 | 79.287794 | 0 | 0 |
| Basic, alpha 1.0 | 63.925090 | 76.781898 | -1.478480 | -2.505896 |
| Full, alpha 1.0 | 63.892235 | 76.772235 | -1.511335 | -2.515558 |

The baseline was recomputed with the pinned repository scorer and exactly matched
the released metrics. Both provisional rerankers performed worse; these results
show no accuracy gain. No additional alpha search, official-dev run or model
selection using official dev was performed. The experiment does not establish
that question-type features generally help or hurt, or that another ridge setting
would behave the same way.

The real runner took 130.118 seconds; the guarded job including tests, process
startup and accounting took 136.884 seconds. Slurm reported peak resident memory
3,177,204 KiB. Outputs include separate targets, both model/binding/prediction
sets and `complete.json`. Compact artifact hashes are recorded below; full
outputs remain outside Git. These measurements apply only to this configuration.

Reproduce with an appropriately sized allocation, the same pinned inputs
and software, a fresh output directory and the explicit opt-in:

```bash
python -m reranker.real_training --manifest /absolute/path/provisional-manifest.json \
  --output /absolute/path/new-run-directory --allow-provisional-provenance
```

All provenance and scientific limitations in the preceding section still apply.
This completes two provisional fits, not independently reproduced BERT training,
team acceptance of the feature/target policy, or final three-system reporting.


## Paired comparison of saved outputs

An output-only comparison reused the saved per-question metrics and prediction
maps; it did not rerun scoring, prediction or fitting. All 9,131 IDs agree across
systems, and all 11 original fit-artifact hashes were unchanged after analysis.

| Change | Raw answers changed | EM correct to wrong | EM wrong to correct | F1 improved | F1 worsened |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline to basic | 1,754 | 474 | 339 | 467 | 948 |
| Baseline to full | 1,770 | 475 | 337 | 465 | 951 |
| Basic to full | 45 | 8 | 5 | 7 | 11 |

Basic and full agree on 9,086 raw answers (99.507%). Full's change relative to
basic is -0.032855 EM and -0.009662 F1 percentage points. Paired EM net counts
reconcile exactly; F1 aggregates agree within 1e-10 percentage points. Raw string
changes may preserve scores because of evaluator normalization or equal partial
credit; unchanged scores do not prove normalized strings are identical.

The eight question-type indicator columns are constant across each question's
candidates. Training-only affine scaling preserves that property, so those
columns cancel under within-question centering and cannot directly change
within-question rankings in this linear objective. The two added question/answer
interactions can vary across candidates. This algebraic observation and the
small basic/full difference do not prove the cause of the regression, establish
statistical significance or justify further tuning. Retain the stronger baseline
while reviewing the provisional policy and paired errors.

## Compact reproducibility identities

SHA-256 values identify bytes, not origin or independently authenticated lineage.
The source revision for the candidate exporter remains
`e56efaa1aacfa5d547286137375ab398baaeefeb`. Archive hashes are listed above.
Full data, predictions, targets and model artifacts are not included in Git.

| Artifact | SHA-256 |
| --- | --- |
| Basic model JSON | `56a16d8fc2a82e64388a36695916f31ec56b869396726600dd3671a5003f313a` |
| Full model JSON | `aedca4827fded431689e580d086f7bb213ac1e9e7315c3832771b1af08c483cf` |
| Basic binding JSON | `f626886b557f89c357bd736fb61c18e4c9e28ef3b072778406eaf11ea6a5da17` |
| Full binding JSON | `422514a2723d51b09bef453992e2778a589bc74c0c41d497fc7ca4f9862d4b07` |
| Complete fit report | `5dbd834e5e0cc768fe0fbe59f1648ba6f1dc41999c78ec7fcb32222c48e02ed1` |
| Paired analysis JSON | `4a9b3ad3abd113ee20667133682c5f1b2907482e2ef4bee22114feb612cdc545` |
| Repository `evaluate-v2.0.py` | `710840ce2c2c61716334c056777032f24660b83eb9869a451036e6a24d8103c4` |
| Repository `scripts/squad_eval.py` | `80df866a5788741632fae24207357b4012a34b39189ba9ad82e3790b1af4d961` |
