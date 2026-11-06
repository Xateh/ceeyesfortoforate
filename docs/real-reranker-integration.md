# Real-label integration — provisional run policy

`python -m reranker.real_training` adds a separate real-label workflow. It does
not weaken `TeamCandidateModel.fit_synthetic`, generate BERT candidates, download
assets or use official dev. Code is implemented; a real run remains blocked by
missing candidate/baseline inputs, training/export lineage review and a separately
bounded real-run allowance.

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
| `validation_baseline` | Independently generated `outputs/bert-base-baseline/internal_val_predictions.json` |

No independent training-split baseline is required: that baseline mode only
supports internal_val/dev. Training first-candidate answers are saved as derived,
not independently verified baseline outputs. Validation IDs and first answers
must match the supplied baseline exactly. Its generation settings must be
explicitly recorded and equal the candidate settings; those declarations and
answer parity are not authentication of the baseline execution.

Full supplied split/source identities are checked before fitting: exact question,
article, context and gold-answer identity; complete union of training-source IDs;
unique IDs/titles; disjoint question/article/exact-context sets across all three
roles; exact gold and candidate Unicode spans; complete candidate coverage for
reranker_train and internal_val. No sampling is used for these gates. This proves
consistency of the supplied package, not that its bytes came from Canvas or that
near-duplicate contexts are absent. The runner consumes assignments; it does not
reshuffle or reimplement the team's split allocation.

Real mode additionally requires hashed checkpoint and tokenizer file descriptors,
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
unchanged template fails closed. A decision reference records approval for this
specific experiment; it is not a claim of team agreement.

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
bytes, not an unavailable official download. No denied retrieval is attempted.

Main's actual `scripts/bert_predict.py` has no max-context filtering: all valid
windows compete by raw logits. Preserve the PR28 traversal order and declared
float32/float64 sum, despite the older main documentation's contrary statement.

## Execution, selection and persistence

After artifact/lineage review and an adequate bounded run allowance, use the
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
is not an authorization to fetch data, fit real inputs or expand task limits.
The caller must apply the approved process memory/runtime/CPU limits; the CLI
does not allocate resources or raise limits itself.

Targets are generated only from reranker-training golds through the team scorer.
For each variant/alpha, the unchanged centered ridge core fits and scales using
training candidates only. Validation predictions precede validation scoring.
Choose highest internal-val F1, then EM, then first configured alpha for exact
ties. Every trial and the selected alpha are recorded. Freeze that selection
before any future official-dev comparison; this runner has no official-dev path.

Outputs include separate targets, derived training first answers, the independent
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

This increment passed **35 selected fixture/regression tests**: eight in
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
that allowance. Set an appropriate bounded run allowance after input sizing.

## Verified release inputs, 6 November 2026

Both compressed team release archives were verified without extraction:

| Release asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `data-v1/data.zip` | 58,488,042 | `bd9577c8dcd7d192c61d2a802e3b8cff10d71e1b1b57cdc5e64bbc4c9e0d245c` |
| `bert-finetuned-v1/final.zip` | 403,707,864 | `a8cf3d7fe9ec5fc07a240353d7d858cc8e4c0dac76bbbee57c982852e8601662` |

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

The archive does not include `run_info.json`, `train_log.jsonl` or
`environment.txt`; the inspected metadata does not establish an exact base
revision. Sean's two candidate exports, matching independent internal-validation
predictions, training/export/base-revision provenance, explicit experiment policy
and a bounded real-run allowance remain outstanding. No real-data fit, BERT
inference, new QA metrics, official-dev run or accuracy gain is claimed here.
