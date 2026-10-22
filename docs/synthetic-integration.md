# Synthetic handoff-to-reranker integration

This local experiment profile joins the actual PR20 draft bundle validator and
feature adapter to PR19's CPU reranker. It does not change those modules or
establish team agreement, real-data correctness or completion of issue #10.
Implementation and review used Codex assistance.

`reranker.integration.SyntheticRun` accepts exactly two assignments:
`reranker_train` and `internal_validation`, each containing `(manifest_bytes,
artifact_bytes_by_filename)`. The bundles keep their original `synthetic` role
and explicit unavailable model/scorer provenance. The assignments are a separate
local execution profile, `cs4248.synthetic-reranker-run/0.1.0-draft`; they are not
production split declarations. Full PR20 validation runs first. Cross-assignment
question IDs, article IDs, exact article hashes and exact context text must be
disjoint. This cannot detect paraphrases or authenticate upstream provenance.

Both assignments must share the exact contract, schema, core, feature spec,
ordered basic5/full15 registry, candidate policy and model/tokenizer provenance
identity. Each retains its own dataset/artifact hashes. `fit()` only accepts the
training assignment; passing validation, BERT-training or official-dev roles
fails before core fitting. Official-dev prediction is also unavailable here.
Scoring the training assignment is permitted for synthetic diagnostics and is
not validation performance. The core receives strict question records and a
separate training-target mapping; gold references never reach it.

```python
from pathlib import Path
from offline_handoff.contract import FILES
from reranker.integration import SyntheticRun

def read_bundle(directory):
    directory = Path(directory)
    return (directory.joinpath('manifest.json').read_bytes(),
            {name: directory.joinpath(name).read_bytes() for name in FILES})

# Supply distinct hand-authored synthetic bundles only.
run = SyntheticRun({'reranker_train': read_bundle('synthetic-train'),
                    'internal_validation': read_bundle('synthetic-validation')})
run.fit()  # alpha=1; no_signal='error', with no implicit fallback
run.save('synthetic-model.json')
run.save_predictions('synthetic-predictions.json')
```

The saved JSON envelope includes both original manifests, their exact byte
hashes, the ordered feature identity, core artifact and its SHA-256. Load with
`SyntheticRun.load(path, assignments)` to replay the same bound bundles. This is
a deliberately narrow replay API: changed validation artifacts also require a
new run, rather than being treated as equal to the training artifacts. Load
checks the wrapper/core feature configuration, target-derived training counts
and error-on-no-signal policy. Hashes detect inconsistent bytes, not malicious
rewrites of both data and hashes or proof of who trained the model. Python callers
are trusted; the wrapper is not a security sandbox.

The core keeps all candidates and all questions in scaling and prediction
coverage, omits single/all-equal groups from ranking loss, and rejects a wholly
no-signal fit. No diagnostic fallback is exposed by this normal-comparison API.
Failed refits preserve the previous model. Predictions are actual model outputs,
with exact supplied answer text and IDs, separate from mapping-illustration.json.

## Evaluator readiness boundary

`reranker.evaluation.preflight_predictions(data_bytes, prediction_bytes)` derives
unique IDs independently from SQuAD-shaped `data/paragraphs/qas/id` records,
rejects missing/extra/duplicate IDs or non-string answers, and preserves Unicode,
whitespace and empty answers without normalization. This is an ID/text preflight,
not full SQuAD data validation. It is suitable before a future supplied evaluator
invocation, whose missing-prediction behavior is unsafe for incomplete mappings.

`invoke_evaluator` performs that preflight before calling an explicitly injected
`runner(source_bytes, data_bytes, prediction_bytes)`. There is no default runner,
subprocess invocation, scorer implementation, source retrieval or no-answer
threshold override. Provenance requires status, source reference, an independently
established expected SHA-256, version (nullable), and the exact invocation revision
`evaluate-v2.0-default-na/0.1.0-draft`. The supplied bytes must match the expected
hash. A future owner must establish that hash independently and authorize a runner
that uses `data_file pred_file` with default no-answer probabilities 0 and threshold
1. The wrapper cannot authenticate that external claim or police arbitrary runner
code. Unresolved/preview-only provenance is refused before dispatch. The verified
supplied-file execution path remains untested and unavailable with current evidence.

Tests inject only a labelled synthetic transport double that returns a sentinel
receipt and computes no metrics. All returned receipts keep
`official_scorer_conformance=false`; a synthetic double is labelled
`synthetic_dispatch_only`. No executed result is claimed to use the supplied scorer.

`tests/fixtures/evaluator-preview-expectations.json` contains six manually derived
expectations based on the preparation worker's 22 October Canvas preview reading.
They cover case/punctuation/articles/space, repeated tokens, empty normalization,
filtering empty golds, multiple golds and the empty fallback. The tests check fixture
labels/structure only, not scorer agreement. Exact original bytes and SHA-256 are
unresolved. The denied download route was not retried; no official file was
reconstructed. These examples are not training labels or official metrics.

## Focused verification

Using the existing Python 3.12.3 / PyTorch CPU environment, run from repository root
with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1:

```sh
python -m unittest discover -s tests -p test_handoff_integration.py -v
python -m unittest discover -s tests -p test_evaluator_boundary.py -v
```

A2/A4/A7/A8/A11 now have focused cross-layer synthetic checks for baseline ties,
actual exact-text predictions and ID coverage, registry/transform and role gates,
no-signal loss/coverage, and bound model replay respectively. The tests assert
actual predictions differ from the fixture illustration, and change equal labels
without changing feature distributions to isolate omitted ranking loss. No real
split, tokenizer, training quality, general held-out inference, scorer conformance
or whole-issue completion is established. The existing aggregate model suite and
BERT smoke tests are outside these focused checks.
