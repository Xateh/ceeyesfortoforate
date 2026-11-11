# Completed context reranker experiments

Run from the repository root on `context-top2-pilot`. No GPU execution is needed
to reproduce the report metrics from the retained scores. The main
`python -m scripts.context_reranker` command trains **one epoch only**.

## Exact five-epoch experiment

[`experiments/span_mean/run.py`](../experiments/span_mean/run.py) is a byte-for-byte
copy of `outputs/context-top2-span-mean-five-epoch-diagnostic/run.py`.
The adjacent `config.json` and `summary.json` are the original saved records,
including the original machine's absolute checkpoint path. Epoch **5** was the
prespecified comparison epoch, not the best epoch selected after calibration.
Seed 4248, batch size 4 questions, AdamW lr 2e-5, default weight decay 0.01,
gradient clipping 1, no scheduler, all 8,751 unequal-F1 pairs per epoch from
12,516 training questions; candidate 2 is chosen iff `score_2-score_1 > bert_gap`.

For archival completeness, the exact original GPU runner can be invoked as below.
**Do not run this for CPU reproduction.** It requires the original RTX 5060 Ti,
the checkpoint/tokenizer directory, the pilot artifacts with the pinned hashes,
and the CLS run's reference orders and summary. Copy it into a fresh directory
two levels below the repository root because it derives paths from its location:

```powershell
New-Item -ItemType Directory outputs/span-mean-replay
Copy-Item experiments/span_mean/run.py outputs/span-mean-replay/run.py
.venv/Scripts/python.exe outputs/span-mean-replay/run.py
```

Required paths for that archival runner:

- `checkpoints/bert-base-baseline/final/` (BERT weights, config and tokenizer)
- `outputs/context-top2-pilot/{weights.pt,metrics.json,predictions.json}`
- `outputs/context-top2-five-epoch-diagnostic/epoch-1-order.json`,
  `epoch-{2,3,4,5}/training-order.json`, and `summary.json`
- The training/validation candidate exports and splits listed below.

Original epoch-5 weights SHA-256:
`c0b3d4397efe9a40c1c3f6647d296981da90d8432158bc892cfea3b1ad615de9`.
CPU reproduction needs the score CSVs instead of these weights.

## CPU reproduction

The original calibration script was not found among this checkout's scripts or
saved experiment runners. `scripts/reproduce_context_results.py` reconstructs the
specified procedure: join scores by question ID, validate candidate/gold identity,
derive candidate F1 labels with the repository official evaluator, exclude equal-F1
training pairs, fit unweighted population means/standard deviations on those
training pairs only, and fit `LogisticRegression(C=1, max_iter=1000)` using
`abs(F1_2-F1_1)` sample weights. Features are `[bert_gap, score_2-score_1]`, preserving
the sign of the correction. The gap-only control uses the same pairs and procedure.
Both models are frozen to JSON before validation; inference reloads the JSON and
chooses candidate 2 iff the decision score is **strictly positive**. No validation
labels are used for fitting or threshold selection.

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r experiments/span_mean/requirements-cpu.txt
.venv/Scripts/python.exe -m unittest discover -s tests -p test_context_reranker.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_context_calibration.py -v
.venv/Scripts/python.exe -m scripts.reproduce_context_results --output outputs/context-reproduction-20261109
```

Use a new `--output` directory on subsequent runs. Input paths can be overridden
with `--exports`, `--splits`, `--scores`, `--dev-candidates`, and `--dev-gold`.
Defaults require these **untracked artifacts**, which must be transferred separately:

- `outputs/issue7-validation-20261104/inference/reranker_train_candidates.json`
- `outputs/issue7-validation-20261104/inference/internal_val_candidates.json`
- `outputs/issue7-validation-20261104/inference/internal_val_predictions.json`
- `data/splits/reranker_train.json`, `data/splits/internal_val.json`
- `outputs/context-top2-span-mean-epoch5-ranking-diagnostic/reranker_train_scores.csv`
- `outputs/context-top2-span-mean-epoch5-ranking-diagnostic/internal_val_scores.csv`
- `outputs/official-dev/dev_candidates.json`, `data/dev-v1.1.json`

The output `checks.json` records input SHA-256 hashes, versions, metrics and any
calibration mismatch; a mismatch also causes a nonzero exit. `frozen_models.json`
stores scaling/model parameters; `bert_dev_predictions.json` stores the rank-one
official-dev predictions. Bulk inputs, predictions and checkpoints stay out of Git.

## Evaluation boundaries

Internal validation has 9,131 questions. BERT's baseline is EM **65.403570** and
F1 **79.287794**. The uncalibrated span-mean epoch-5 model gives EM **65.414522**,
F1 **79.295613**, 3 changed answers, 1 fix and 0 breaks.
The reconstructed two-feature calibration matches the requested target exactly:
344 changed, 81 fixes, 78 breaks, EM **65.436425**, F1 **79.339035**.
The gap-only control gives 135 changed, 26 fixes, 28 breaks, EM **65.381667**,
F1 **79.285790**. No calibration mismatch was found.

BERT's **official dev** scores, computed on CPU from rank-one answers in
`outputs/official-dev/dev_candidates.json` against `data/dev-v1.1.json`, are
EM **80.018921** and F1 **87.553216**. All **10,570** unique gold question IDs are
covered exactly, with no missing or extra IDs. The export metadata identifies
`checkpoints/bert-base-baseline/final`, float32 logits, n-best/max candidates 20,
max answer length 30, and first-occurrence baseline tie breaking. No separate
saved official-dev evaluation result or predictions file was found; these scores
are recomputed from the saved candidate export, not read from an earlier score log.

Scoring uses the checked-in `evaluate-v2.0.py` via `scripts/squad_eval.py`, the same
official evaluator used for the original experiment. SQuAD 1.1 has no unanswerable
questions; no no-answer threshold is applied. These official-dev BERT scores are
separate from all reranker internal-validation results; no reranker official-dev
score is claimed. Exact equivalence to a separately supplied course evaluator and
the upstream BERT training lineage remain unverified, as in `docs/reranker.md`.

Original experiment environment: Python 3.12, torch 2.8.0+cu128,
transformers 4.57.1, tokenizers 0.22.2, numpy 2.5.4. CPU reconstruction used
Python **3.12.10**, the same torch/transformers/numpy versions, scikit-learn
**1.7.2**, scipy **1.18.1**, joblib **1.6.0**, threadpoolctl **3.8.0**, and
cloudpickle **3.1.2**. Scikit-learn/scipy were installed for this reconstruction;
the original calibration environment was not available. The default lbfgs solver,
intercept, tolerance and class weighting were retained.

Verified locally on 2026-11-09: **5 pooling tests and 3 calibration tests passed**;
the full CPU reproduction command exited successfully. The official evaluator CLI
also independently returned the same dev EM/F1:

```powershell
.venv/Scripts/python.exe evaluate-v2.0.py data/dev-v1.1.json outputs/context-reproduction-20261109/bert_dev_predictions.json
```

Small verification records are retained in
[`cpu-checks.json`](../experiments/span_mean/cpu-checks.json), including input
hashes and full-precision metrics. Frozen calibration parameters are in
[`frozen-calibration.json`](../experiments/span_mean/frozen-calibration.json).
No required CPU input is missing on this PC. A teammate still needs all separately
listed untracked artifacts; Git contains neither the data nor the score CSVs.
