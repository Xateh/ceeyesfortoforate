# CS4248 G19 code submission — 11 November 2026

This package reproduces the BERT/basic/full all-candidate comparison and the experiments cited by the latest available report. Read RESULTS.md for tables, REPORT-AUDIT.md for report changes still needed, and provenance/reproduction-notes.md for limitations. The submission deadline supplied by the team is **12 November 2026, 23:59 Singapore time**.

## Files to retain together

- **G19-CS4248-code.zip**: source, configs, dependencies, unmodified course evaluator, frozen linear models/bindings, results/predictions, context score CSVs, original contextual predictions and report snapshot.
- **G19-CS4248-data.zip**: exact existing data.zip, inference.zip and dev_candidates.zip under assets/. Includes original SQuAD data, article splits, tokenized data and candidate exports.
- **G19-CS4248-checkpoints.zip**: exact final.zip BERT checkpoint release plus original contextual pilot and epoch-5 CLS/span-mean weights. Needed for BERT regeneration or contextual model inspection, not CPU score replay.

Extract all desired archives into the **same new directory**. Entries are relative to its root. Do not place ZIPs in Git. Course instructions require supporting code and data, so retain/provide the companion archives with the code handoff; no upload-size exemption was found. This package does not submit anything. SHA256SUMS.txt beside the ZIPs identifies their exact delivered bytes. provenance/assets.json identifies the exact release tags, URLs, sizes and hashes. The dev export has no public release URL and is supplied directly.

## Setup and full verification

Python 3.12.10 on Windows was used; run commands from the extracted root. Create a new environment using the verified Windows commands below.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-replay.txt
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -X utf8 -m scripts.prepare_submission_inputs
python -X utf8 -m scripts.run_submission_tests --output verification/tests
python -X utf8 -m scripts.replay_submission --output verification/replay
```

Main inference, linear fitting and the full results replay need CPU PyTorch in addition to requirements-replay.txt: reranker/core.py imports torch. The full test suite also imports transformers. Replay never loads BERT weights or uses a GPU. Allow roughly 4 GiB RAM for linear fitting; retained tokenized data and optional checkpoints require several GiB disk. Replays require a new output directory on each run. Each test suite uses a separate process as required by the contract import-boundary tests. Avoid Python `-O`, which disables result assertions.

The retained frozen models use Windows CRLF bytes. Their strict reload check compares the saved model with platform-default reserialization, so LF reserialization on Linux/macOS fails that check even when parameters are identical. Windows is the verified target for this frozen handoff; cross-platform serialization needs a separately validated change before claiming Linux/macOS replay support. Do not change artifact bytes or bypass their hash checks. The original ZIPs remain unchanged; this review clarification corrects their README's minimal-dependency and cross-platform implications.

For Git review, commit 1bb3e15 restores the two model files' exact verified bytes after Git normalized their line endings in 16ccbfb. Parameters, source bindings and reported metrics are unchanged. Large inputs, retained score CSVs, original predictions and generated result maps are supplied in the existing ZIPs rather than Git: extract the code and data ZIPs together for full replay. RESULTS.md and provenance/ contain the tracked result and provenance records.

The replay checks all six main prediction maps against the packaged maps, every exact ID set (9,131 internal / 10,570 dev), target metrics, fixes/breaks, original basic-model parity, course/repository per-question scoring parity, context and no-length predictions, oracle, and the separately labelled gated rule. It writes all commands and receipts. Windows course evaluation uses `-X utf8` because the original course script uses platform-default file decoding. Never edit/normalize pinned source or model bytes without creating new bindings.

## Individual all-candidate inference and evaluation

```powershell
python -X utf8 -m scripts.submission_compare --split internal_val --candidates inputs/inference/internal_val_candidates.json --gold data/splits/internal_val.json --original-baseline inputs/inference/internal_val_predictions.json --output outputs/check-internal
python -X utf8 -m scripts.submission_compare --split dev --candidates inputs/official-dev/dev_candidates.json --gold data/dev-v1.1.json --validation-receipt outputs/check-internal/metrics.json --output outputs/check-dev
python -X utf8 course/evaluate-v2.0.py data/dev-v1.1.json outputs/check-dev/full-predictions.json
```

The inference function accepts no gold data. All three maps are saved before evaluation opens gold. Basic/full score every exported candidate, retaining the first in exact ties. Models, feature order, scaling and alpha are frozen; dev labels never change them. Default model directory is frozen/linear and independently retained binding pins are in configs/submission-models.json. The original frozen-basic-model.json is preserved separately; scripts/final_comparison.py implements the different gated top-two rule, not main top-20 inference.

## Reproduce ridge training

With the checkpoint companion extracted (its archive is hashed, not loaded), run:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
New-Item -ItemType Directory -Force outputs
python -X utf8 -m reranker.real_training --manifest configs/reproduction-manifest.json --output outputs/linear-refit --allow-provisional-provenance
```

The manifest uses relative paths, actual input/archive hashes, the real article splits and the original scorer semantics, with alpha grid [1.0]. Its explicit provisional lineage records missing original training logs/environment; do not replace them with invented evidence. This deterministic CPU fit reproduces both model states and the specified internal scores. The original recovered artifacts remain under originals/, configs/frozen-basic-model.json and assets/; new fits are under results/linear-training. The saved manifest in that result directory is a historical receipt; use configs/reproduction-manifest.json as the runnable manifest.

## Data preparation and BERT regeneration

```powershell
python -X utf8 -m scripts.prepare_submission_inputs --checkpoint
python -X utf8 scripts/split_data.py --verify
python -X utf8 scripts/preprocessing.py
python -X utf8 scripts/bert_train.py --output-dir outputs/new-bert-training
python -X utf8 scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split dev --candidates --batch-size 64 --max-answer-length 30 --output-dir outputs/new-official-dev
```

The first command already supplies tokenized data and the saved final checkpoint, so preprocessing and BERT training are unnecessary for this handoff. They document the original pipeline, configs/bert-qa.json and configs/bert-train.json. GPU BERT execution needs the compatible PyTorch CUDA build (the original RTX 5060 Ti used torch 2.8.0+cu128). No GPU regeneration was run for this package; the exact dev export is supplied and hash-checked. A new hardware/software export may have a different hash: retain it separately and investigate instead of silently replacing the frozen export. Check the scripts' --help for alternate data/output paths. BERT optimizer seed is 42; article split seed is 4248.

## Context, ablation and oracle

scripts/replay_submission.py gives the tested portable CPU commands for every report row. Context calibration uses inputs/context-scores/{reranker_train,internal_val}_scores.csv, training-only scaling, F1-difference sample weights, C=1 and max_iter=1000. The two models are saved before validation. CLS epoch-5 is rescored from its retained original predictions; span-mean and calibration are regenerated from retained scores. Bootstrap settings are explicitly 20,000 article draws, seed 4248; see the report interval discrepancy in REPORT-AUDIT.md.

Original five-epoch GPU runners/configs/orders are preserved under originals/context and experiments/span_mean for provenance. They retain historic paths and hardware assertions and are **not** the portable CPU entry points. All local-only inputs needed for the tested replay are supplied. The one-epoch scripts/context_reranker.py is not equivalent to the five-epoch report experiment. Contextual weights are supplied separately for inspection/further inference; CPU score replay does not claim to retrain those encoders.

## Credits, scope and human items

Authors listed in the report: Feng Jiaqi, Goo Zheng Xien Sean, Leu Tho Anh Duy, Teh Xu An and Trinh Hoang Hai. Preserve the team's actual attribution: Hai's data/BERT/evaluator pipeline; Sean's candidate exporter; Xu An's ridge core, basic/full integration, persistence and validation. The repository and experimental branch history are recorded in provenance/branches.json. Other assignments must be confirmed by the team rather than inferred. Codex assisted implementation, tests, review and documentation, including this packaging pass. Each member must supply/review their own truthful AI-use declaration; a recovered form is included in course/.

This is the code/data handoff, not a finished report, slides, recording or set of declarations. The latest report still has placeholders and lacks the completed main official-dev comparison. No files were uploaded to Canvas, no release published and no remote branch pushed. Older project guides are retained for history; this README and RESULTS.md describe the completed submission replay.
