# Label-free held-out inference (synthetic software profile)

`reranker.heldout.HeldoutRun` fits on one labelled synthetic PR20 bundle and
accepts independently hashed, previously unseen inference bundles. The existing
`SyntheticRun` API and its exact two-bundle replay remain unchanged. This adds
software coverage for new batches, not real QA performance or issue #10 completion.
Implementation and review used Codex assistance. All schema/feature choices remain
provisional software contracts, not team agreement.

## Separate input boundaries

Training uses the unchanged `cs4248.offline-bundle/0.1.0-draft` bundle: targets and
label audit remain separate from question features. `fit(role='reranker_train')`
is the only permitted fit role. Scaling is learned only from training; singleton
and equal-target groups remain in scaling but contribute no ranking loss. An
entirely no-signal batch fails; this API exposes no diagnostic fallback. Failed
refits preserve the previous fitted model.

Inference is a distinct `cs4248.label-free-inference/0.1.0-draft` manifest with
exactly these fields:

- `schema_id`: the new label-free schema ID above.
- `status`: `synthetic_illustration_not_production_valid`.
- `execution_role`: `internal_validation` only. The role is a synthetic local
  execution assignment; the split artifact retains its original `synthetic` role.
- `dataset_id`: the new batch's nonempty identifier.
- `identity`: exact training identity fields: `schema_id` (the original labelled
  schema ID), `contract_id`, `feature_spec_id`, ordered `feature_registry`,
  `candidate_policy`, `core_commit`, `generator_commit`, `feature_code_commit`,
  `checkpoint_sha256`, and `tokenizer_id_and_revision`.
- `artifact_sha256`: exact SHA-256 values of the five supplied artifact byte strings.

The artifact dictionary must contain exactly `contexts.json`, `candidates.json`,
`questions.json`, `provenance.json`, and `split-manifest.json`. No targets, gold
audit, illustration mapping or undeclared file is allowed; extra fields inside
records/manifests also fail. The validator never synthesizes missing labels or
calls the labelled bundle validator for inference. It checks hashes, exact spans,
feature formulas/order, baseline-first deduplication, complete origins, and split
coverage using the existing offline validation functions. The exact basic5/full15
registry and all question-type exclusions from basic are preserved.

The new batch must match the saved training identity, but its own dataset ID,
manifest bytes and artifact hashes may differ. Reject overlap with training by
question ID, article ID, exact complete-article hash, context ID or exact context
text hash. Hashes do not detect paraphrases or prove that supplied articles are
complete/authentic. This synthetic profile has no global BERT-train/official-dev
registry; those execution roles and all real production status are refused.

## Persistence and use

```python
from reranker.heldout import HeldoutRun

# Each bundle is (manifest_bytes, {filename: artifact_bytes}).
# Training is labelled; unseen_batch is the five-file label-free profile.
run = HeldoutRun(training_bundle).fit()
run.save('synthetic-model.json')
loaded = HeldoutRun.load('synthetic-model.json', training_bundle)
predictions = loaded.predict(unseen_batch)
receipt = loaded.save_predictions('synthetic-predictions.json', unseen_batch)
```

Reload intentionally requires the original labelled **training** bundle. It
revalidates that bundle and verifies its original manifest/hash, identity and
training exclusion index, then restores the core without fitting. No original
validation bundle is required. A deployment artifact that reloads without
training inputs is outside this bounded change. Neither loading nor inference
uses held-out targets/gold; training labels are checked during loading only.

The `cs4248.synthetic-heldout-reranker/0.1.0-draft` model envelope binds the exact
training manifest/hash and exclusion index, feature order/group/spec, checkpoint/
tokenizer and candidate policy. It stores the core's fitted preprocessing, ridge
configuration/state and SHA-256, checks no-signal policy and training counts, and
rejects incompatible bindings. New inference calls do not update learned state.
Exact score ties choose the first validated candidate. `predict` returns every
question ID with one exact supplied candidate answer; no text normalization.

`save_predictions` writes that actual mapping and returns a receipt containing
the training-manifest, model, new inference-manifest, all input artifact and
prediction-byte hashes, role and question count. Retain the returned receipt
separately if required; it is not a score/metric or an illustration mapping.
Both model and receipt remain explicitly non-production synthetic artifacts.
Callers choose output paths; like the existing core API, writes replace those
paths. Hashes check consistency, not authenticity or malicious coordinated
rewrites of model and hashes. Python callers are trusted, not sandboxed.

Checkpoint/tokenizer/generator/feature-code identities remain the original
`unavailable:synthetic-no-model-or-scorer` sentinels. No trained QA checkpoint,
real tokenizer/data, official evaluator bytes or conformance is supplied here.
Unknown or production provenance fails closed. No scorer implementation or
retrieval is added. These gates must not be removed to promote fixtures to real
training inputs.

## Focused verification

`python -m unittest discover -s tests -p test_heldout_inference.py -v` exercises
both feature groups on two newly hashed synthetic batches after reload. It checks
frozen training transforms, dedup/baseline ties, exact Unicode predictions and
IDs, omitted ranking loss for no-signal groups via existing regression coverage,
forbidden labels/roles/overlap, malformed spans/features, provenance/serialization
changes and failed-refit preservation. Existing replay/core/handoff/evaluator
suites remain separate regression commands; run the offline contract suite in a
fresh process because it checks that no model modules were imported.

Use the existing CPU environment with OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS set to 1, CUDA_VISIBLE_DEVICES empty, and a bounded resource
runner. This task retains the stricter existing 1 GiB address-space limit, pins
one CPU, sets PyTorch intra/inter-op threads to 1 in model test processes, and
tracks cumulative test wall time against the approved 600-second cap. Exact
commands, peaks and outcomes are recorded outside the worktree in task evidence.
