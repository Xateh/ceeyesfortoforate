# Provisional SQuAD train preparation

This local-only pilot prepares and audits data. It does not train a QA model,
generate candidates, produce scorer labels, evaluate accuracy, or establish team
agreement. The existing reranker and synthetic interfaces are unchanged.

The only accepted source bytes are pinned in `squad_preparation.ASSET_HASHES`:

- `rajpurkar/squad`, revision `7b6d24c440a36b6815f21b70d25016731768db1f`,
  configuration `plain_text`, `plain_text/train-00000-of-00001.parquet`.
- `google-bert/bert-base-uncased`, revision
  `86b5e0934494bd15c9632b12f734a8a67f723594`: `tokenizer.json`,
  `tokenizer_config.json`, `config.json`, and `vocab.txt` only.

The caller supplies these five files locally. The program has no downloader,
remote-code loader, model loader or official-dev input. Each file must match its
SHA-256 before processing. The train hash was checked against the pinned Hub LFS
OID; tokenizer/config bytes against their pinned Git blob OIDs. Their actual
SHA-256 values are also in every successful run's provenance. A changed source
requires deliberate code/provenance review, never an implicit revision fallback.
The tokenizer JSON is self-contained; its original normalizer and postprocessor
are used directly by the installed `tokenizers` library, without Transformers or
PyTorch imports. `config.json` describes the base architecture, not trained QA
checkpoint provenance.

## Split and pilot choices

The draft identifier is `cs4248.squad-preparation/0.1.0-draft`, seed 42.
Source `title` defines article identity; SHA-256 of the exact title is its ID.
All rows of one article stay together. Articles sharing any exact context are
joined transitively before assignment. Sorted connected components are shuffled
with Python `random.Random(42)`, then assigned using floor cut points at 80%
and 90%. This is an 80/10/10 proposal by article components, not by question
counts. The manifest reports actual counts and maximum component size.

The roles are `bert_train`, `reranker_train`, and `internal_validation`; all come
from the official training partition. The source must have at least ten article
components. Duplicate question IDs, unexpected schema and cross-role article or
exact-context reuse fail. Two streaming source passes must produce equal
manifests. Synthetic tests also reverse input order and check equality.
Article titles are source identities, not independently verified Wikipedia page
identities. Exact-context hashing cannot establish absence of semantic or near
text duplicates. No official-dev comparison is performed.

The default pilot selects 800/100/100 question IDs after sorting and independently
seeded shuffling within each role. Selection precedes alignment and does not use
gold quality. Smaller limits from 1 to 1,000 use corresponding floor cut points.
A question is never replaced because its alignment is inconvenient.

## Alignment and outputs

Pairs are `(question, context)`, with only the context truncated, maximum length
384 including special tokens, stride 128, and no padding. Question capacity must
exceed the stride. Context token positions are identified by sequence IDs;
question/special-token offsets never stand in for context offsets.

Every sampled gold must equal the original context slice at its supplied
zero-based Unicode code-point half-open offsets. No normalization, searching for
another occurrence, trimming or widening is used. All overflow windows and all
annotated golds are audited. An exact token span must reproduce both character
boundaries and the original text. This is offset-based reconstruction, not a
claim that decoding uncased WordPiece IDs reproduces original text.

Window/gold classifications are `exact`, `outside`, `partial`, or
`unrepresentable_boundary`. A full untruncated context audit distinguishes
nonrepresentable boundaries from truncation. An answer longer than a context
window, or longer than the overlap and crossing windows, receives an explicit
coverage exception. Other missing coverage raises an unexplained-alignment error.
Exceptions remain in the sample and audit; there is no fabricated training span
or silent dropping. A run with exceptions is an audit result, not a ready-made
supervised training package. Choosing a future training policy requires review.

Successful output, outside Git:

- `split-manifest.json`: all source question/article/context identities and roles.
- `selection.json`: ordered pilot IDs, selected independently of gold quality.
- `pilot-contexts.json`: exact original sampled contexts keyed by SHA-256.
- `pilot-inputs.jsonl`: IDs, question text and token windows; no gold annotations.
- `pilot-gold-audit.jsonl`: separate gold references, token alignment and coverage.
- `audit-summary.json`: counts, exceptions and reproducibility/disjointness results.
- `provenance.json`: pinned identities, actual hashes, packages and code hash.
- `artifact-hashes.json`: SHA-256 of the other generated artifacts.

The pipeline refuses an existing output directory. A processing error preserves
partial outputs and attempts a failure receipt without masking the original
error if storage itself fails. Accept a run only with exit 0, no failure receipt,
a complete audit summary and the final artifact-hash inventory. These artifacts are not PR20 handoff bundles and cannot imply
that real reranker labels or general real-data execution are enabled.

## Bounded execution

Use the existing environment; do not install packages. The pilot was authorized
for one CPU/thread, CUDA disabled, 1 GiB memory, 600 cumulative execution seconds,
64 MiB download bodies and 512 MiB new storage. Data/tokenizer assets stay outside
Git. The CLI requires one-CPU affinity, `RLIMIT_AS` at most 1 GiB, and these values:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=''
python squad_preparation.py --assets /path/to/local-assets \
  --output /path/to/new-output --limit 1000
python -m unittest discover -s tests -p test_squad_preparation.py -v
```

Set affinity and limits in the launcher before either command; use a cumulative
budget watchdog for time/storage. The CLI checks process resources but is not a
replacement for download/storage accounting. PyArrow streams 256 rows per batch
with threaded reads disabled. No unrelated numerical/model suite is required for
this separate preparation module. The supplied evaluator's exact bytes/hash and
executable conformance remain unresolved; this pilot does not access it.

## Verify an existing preparation receipt

`preparation_receipt.verify_receipt(directory, expected_inventory_sha256)` and
its CLI validate saved outputs without opening source assets, tokenizing,
loading a model or executing a scorer. The verifier uses only the standard
library. Keep the producer's successful exit receipt and the SHA-256 of
`artifact-hashes.json` in a separately trusted record; supply that retained hash:

```sh
python preparation_receipt.py --receipt /path/to/existing-output \
  --inventory-sha256 "$TRUSTED_PREPARATION_INVENTORY_SHA256"
python -m unittest discover -s tests -p test_preparation_receipt.py -v
```

Computing a new expected hash from an untrusted bundle does not authenticate it.
The verifier checks the caller's pin, then the exact seven-artifact hash map,
regular-file membership, strict JSON and a 64 MiB total input-size ceiling.
Missing/extra files, symlinks, failure receipts, duplicate JSON keys and nonfinite
numbers fail. Use an immutable local snapshot; concurrent filesystem writers
are outside this interface. The execution and memory limits above still apply.

It checks the pinned draft, producer-code hash and declared source/tokenizer
identities; reconstructs article components and seed-42 role assignment from the
manifest; verifies counts and exact ordered pilot coverage; and checks context
hashes, role identities, label-free input keys and token-window domains. Saved
offsets must reproduce gold text and per-window alignment classifications.
Summary counts and retained exception lists must agree with those records.
No fixed 997/3 success threshold is imposed. Run this test file separately from
suites importing optional packages; one test confirms its standard-library-only
import boundary.

A successful result says `verified_preparation_receipt` and explicitly sets
`training_readiness_established` to false. It proves consistency with the supplied
pin, not authenticity of that pin, actual source execution, tokenizer-to-token-ID
conformance, unseen source completeness or the truth of a recorded successful
exit. Full untruncated context offsets are not saved, so their classifications
are checked for structural consistency but cannot be independently recomputed.
The two-pass reproduction flag is a producer assertion; this verifier recomputes
only the recorded manifest's split. Package versions are recorded nonempty
strings, not an environment compatibility guarantee. The pinned producer hash is
intentional: changed producer bytes require deliberate verifier review.

This preserves all alignment exceptions. It makes no decision about dropping,
widening or relabelling them, no team split agreement, no official scorer claim
and no supervised-training readiness claim. Synthetic tests are hand-authored
receipts with toy token IDs, not tokenizer outputs or scorer results.
