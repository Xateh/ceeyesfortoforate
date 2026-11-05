# Adapter-aware PR28 model persistence (synthetic profile)

`reranker.team_model.TeamCandidateModel` closes the specification/provenance gap
left by using a standalone core model after PR28 conversion. The existing core,
team adapter and older SyntheticRun/HeldoutRun APIs remain unchanged. New code
must choose this wrapper to obtain its binding checks; the raw core is not changed.

```python
model = TeamCandidateModel.fit_synthetic(
    train_export_bytes, separate_targets, group="basic",
    target_status="synthetic_hand_authored_not_scorer_outputs",
)
pin = model.save(new_model_path)       # exclusive creation; keep pin separately
loaded = TeamCandidateModel.load(
    new_model_path, train_export_bytes, expected_sha256=pin,
)
answers, receipt = loaded.predict_with_receipt(new_internal_val_export_bytes)
```

These are in-memory JSON-byte interfaces. The example assumes tiny synthetic
fixtures, not real labels or an approved team target policy. The status parameter
is a caller attestation, not an origin detector. No scorer or BERT is executed.
Basic/full definitions remain `candidate-features/0.1.0-draft`, not team agreement.

## Saved contract and reload

Schema: `cs4248.pr28-reranker-model/0.1.0-draft`; status:
`synthetic_fit_not_production_valid`. Exact envelope fields are `schema_id`,
`status`, `binding`, `target_status`, `training_targets_sha256`, `model`,
`model_sha256`. Unknown fields and unsupported identities fail.

Binding retains the adapter profile, candidate schema version, inspected producer
revision, core commit, exact ordered feature registry/spec/group, and relevant
shared generation declarations: checkpoint path, maximum answer length, search and
candidate limits, logit dtype, tie/max-context policy, NumPy/PyTorch versions and
device. It also retains the complete training metadata, exact training export hash
and question/article/exact-context exclusion index. Export paths are declarations;
checkpoint/tokenizer content hashes are absent and are explicitly not verified.

Load requires an independently retained SHA-256 of the whole model file. A hash
inside that file alone is insufficient. It validates the pin before parsing,
reconstructs the binding from original label-free training bytes, validates the
exact feature configuration against the running adapter, and checks core state
hash/configuration/counts through the existing strict core loader. Schema/type,
equal-width order/spec/group changes, diagnostic fallback and malformed numeric
state are refused. No fit or preprocessing update occurs at reload.

Targets are supplied only to fitting, with the core's exact coverage/domain and
no-signal guardrails. Target values/golds are absent from saved envelopes and all
reload/inference inputs. The target digest is SHA-256 of `contract.encode(targets)`
(UTF-8, ensure_ascii=False, indent=2, allow_nan=False, trailing newline), not a claim
about an original target file whose bytes were never supplied. Object insertion
order affects that digest. It records the original supplied mapping serialization.

## New batches and integrity limits

Reload needs the exact original TRAINING export, but inference does not require a
previous validation export. New label-free internal_val batches may have different
question IDs, contexts and input-file paths. Each is independently hashed and
adapted; role, declared generation identity and training exclusions are checked.
Exact question/article/context overlaps fail. No validation labels are accepted,
and preprocessing never changes. Team candidate/tie/dtype semantics are inherited
unchanged from the PR28 adapter. The receipt includes current batch, prediction,
core-state and adapter-identity hashes, with origin_authenticated=false.

Model files are created exclusively, including refusal of existing symlink paths.
Predictions/receipts are returned in memory; callers choose new output paths. A
failed classmethod fit creates no replacement for an existing model instance.

Hashes protect byte identity relative to a trusted separately retained pin; they
cannot establish origin. Jointly replacing a file AND its trusted external pin
changes the trust anchor and is not detectable by this wrapper. Synthetic status,
checkpoint paths and producer revision are not proof of real training. The
producer ref identifies inspected software, not a verified generator execution.
Merged-split membership, tokenizer/checkpoint byte provenance, target policy and
real evaluator conformance remain outside this synthetic implementation.

## Verification

Run `python -m unittest discover -s tests -p test_team_model.py -v` under the shared
600-second ledger, one CPU/thread, CUDA disabled and 1 GiB cap. The suite uses the
existing invented PR28 fixtures. It checks both groups, two unseen batches,
round-trip scores/selections and frozen state with fit forbidden during reload;
exact score ties before/after reload, including reversed supplied candidate order;
external-pin and semantic tampering, incompatible feature order/spec/identity,
role/overlap/label rejection, core guardrails and existing-artifact preservation.
No real data or model generation is needed. Keep the existing team-adapter,
core and held-out regression suites in separate processes when rerunning them.
