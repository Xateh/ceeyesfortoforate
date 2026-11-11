# Frozen official dev comparison

This comparison evaluates a small, fixed set of extractive QA systems on the
10,570 questions in the original SQuAD v1.1 development set. Feature and alpha
selection use the earlier internal-validation experiments. Official-dev scores
are used for reporting, not for another configuration search.

## Frozen systems and selection

All systems use the same BERT top-20 candidate pool, maximum answer length 30,
top-20 start/end search, raw start-plus-end logits, and existing window/tie rules.
The rerankers were trained on `reranker_train` and are not refitted on official dev.

| System | Role | Alpha | Internal EM | Internal F1 |
| --- | --- | ---: | ---: | ---: |
| BERT | Baseline | — | 65.404 | 79.288 |
| Basic without length | Main reranker | 0.1 | 65.272 | 79.256 |
| Full without length | Main question-aware reranker | 0.1 | 65.250 | 79.234 |
| Basic with raw length | Controlled length ablation | 0.1 | 65.710 | 78.575 |

The main basic system retains start/end logits, the digit flag, and the uppercase
flag. The full system adds the existing question-type indicators and the two
question–answer interactions. Standalone question indicators cancel within each
question; the interactions can change rankings.

Training and evaluation both use 20 candidates for a matched final comparison.
Within the basic-without-length and full-without-length profiles, alpha is chosen
by highest internal F1, then EM, then the first configured alpha. These profiles
had higher selected internal F1 than the other feature-rich profiles in their
respective families. The logits-only control is excluded from the main systems
because it essentially reproduces BERT. The extra raw-length system holds alpha
fixed at 0.1 so its comparison with basic without length isolates feature removal.

The systems, source model hashes, checkpoint files, source data, tokenized data,
generation settings, and selection results were pinned before local official-dev
inference in [official-dev-frozen.json](../configs/official-dev-frozen.json).
Its SHA-256 is:

```text
b3295d4cde924720064c04d8c96f3373065e0c6141f00125e62c11dce41c541b
```

The [internal-validation report](reranker-experiments.md) retains the broader
feature ablations and alpha search. Only these four systems are evaluated here.

## Results

Evaluation completed on all **10,570 official-dev questions**. Every result is from
the same newly generated BERT candidate pool. All rerankers use their frozen
weights and alpha 0.1. Scores are percentages; changes are percentage points.

| Frozen system | EM | F1 | EM change versus BERT | F1 change versus BERT |
| --- | ---: | ---: | ---: | ---: |
| BERT baseline | 80.019 | 87.553 | +0.000 | +0.000 |
| Basic without length | 79.943 | 87.524 | -0.076 | -0.029 |
| Full without length | 79.991 | 87.528 | -0.028 | -0.025 |
| Basic with raw length ablation | 80.284 | 87.198 | +0.265 | -0.355 |

The baseline reproduces the previously documented rounded result of **80.02 EM /
87.55 F1**. The two main rerankers are close to BERT but slightly below it on both
metrics. The raw-length ablation increases EM while decreasing F1; it does not
establish an improvement across both metrics.

### Paired exact answer changes

| Reranker | BERT EM errors corrected | Correct BERT answers broken | Net exact answers |
| --- | ---: | ---: | ---: |
| Basic without length | 50 | 58 | -8 |
| Full without length | 50 | 53 | -3 |
| Basic with raw length ablation | 301 | 273 | +28 |

Basic without length changes 397 answer texts relative to BERT; full without
length changes 409. The full model differs from basic on 84 answers, correcting
10 EM errors and introducing five. Its additional gain over basic is only
**0.047 EM / 0.004 F1 points**.

### Controlled length ablation

Removing raw length while holding alpha and the candidate pool fixed changes
official-dev scores by **-0.341 EM / +0.326 F1 points**. Thus, the observed F1
benefit of removing length transfers in direction to official dev, with an EM
cost. This is more precise than saying length is always harmful: at the matched
alpha 0.1, raw length trades some partial-answer overlap for more exact answers.
The earlier large alpha-1 losses belong in the internal-validation ablation table,
not in this matched-alpha final comparison.

### Uncertainty across articles

A paired bootstrap resamples the 48 dev articles with replacement, retaining all
questions in each sampled article. Each of 5,000 resamples recomputes the
question-weighted metric difference, using seed 4248. The table gives descriptive
95% percentile intervals in percentage points.

| Comparison | EM difference and 95% interval | F1 difference and 95% interval |
| --- | ---: | ---: |
| Basic without length versus BERT baseline | -0.076 [-0.251, +0.096] | -0.029 [-0.139, +0.073] |
| Full without length versus BERT baseline | -0.028 [-0.211, +0.148] | -0.025 [-0.141, +0.084] |
| Basic with raw length ablation versus BERT baseline | +0.265 [-0.525, +1.042] | -0.355 [-0.807, +0.069] |
| Basic without length versus Basic with raw length ablation | -0.341 [-1.059, +0.420] | +0.326 [-0.091, +0.776] |
| Full without length versus Basic without length | +0.047 [-0.029, +0.133] | +0.004 [-0.046, +0.052] |

All displayed intervals include zero. The observed differences therefore do not
provide robust evidence of a general improvement under this article-resampling
analysis. These are descriptive intervals for the fixed comparison, not a claim
that official dev was previously untouched or a correction for earlier feature
exploration. No additional configurations were chosen from these results.

### Findings for the project report

The fine-tuned BERT baseline achieved 80.019 EM and 87.553 F1 on official dev.
Rerankers selected using internal-validation F1, with raw span length removed,
achieved 79.943 EM / 87.524 F1 for basic and 79.991 EM / 87.528 F1 for the
question-aware model. Neither improved the baseline. The matched raw-length
ablation achieved 80.284 EM / 87.198 F1, demonstrating an EM–F1 trade-off rather
than an across-metric gain. Together with the internal ablations, these results
show that the current linear reranker's behavior is sensitive to length features
and regularization, while the existing question–answer interactions add little.

Absolute internal and official-dev scores should not be interpreted as a gain
caused by the reranker: they use different questions and gold annotations. The
relevant comparisons are against BERT within each dataset. Keep all internal
experiments in the development-results section and this fixed four-system table
in the final evaluation section, including its negative results.

## Saved artifacts and verification

- [Final results CSV](results/official-dev/results.csv) and
  [complete run record](results/official-dev/complete.json).
- [Paired changes and article bootstrap intervals](results/official-dev/paired-analysis.json).
- Four `*-predictions.json` files in `outputs/official-dev-final/`, each covering
  every official-dev question exactly once.
- `outputs/official-dev-candidates/dev_candidates.json` (local cached dev candidate export),
  shared by all systems, and a copy of the frozen configuration in the results folder.
- The numerical source, model, data, candidate, and prediction hashes are retained
  for reproducibility. All frozen model hashes and fitted states stayed unchanged.

The new official-dev suite passed six focused tests covering role validation,
source overlap, label-independent adaptation, model/feature pins, inference without
fitting, and the frozen CLI. The ten existing span-selection tests also passed.
Small result summaries are versioned under `docs/results/`. Generated data,
weights, candidates, and prediction files remain ignored by Git; share those
artifacts separately when distributing the source and report.

## Evaluation history and interpretation

BERT official-dev scores already appeared in [the repository evaluation
documentation](eval.md) before this comparison. This is a limited comparison
whose configurations were frozen before the new local evaluation; official dev
must not be described as a previously untouched test set. Internal validation
was also used for BERT validation and extensive reranker exploration.

The local BERT checkpoint is pinned by file hashes. Original upstream training
logs and the checkpoint-to-training-export lineage remain unverified. Hashes
identify the local bytes used, rather than independently proving their origin.

## Reproduction

Use the existing project environment and downloaded artifacts, plus the retained
selected model files and original `outputs/reranker-exploration/` run records
referenced by the frozen configuration. The committed summaries under
`docs/results/` support review; they do not include the fitted model files. The output
directories must be new for a fresh comparison. To reuse the cached BERT export,
skip the first command and choose a new evaluator output directory.

```bash
source .venv/bin/activate
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python scripts/bert_predict.py \
  --checkpoint final --split dev --candidates \
  --max-answer-length 30 --batch-size 8 --log-every 20 \
  --output-dir outputs/official-dev-candidates

OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python scripts/evaluate_official_dev.py \
  --freeze configs/official-dev-frozen.json \
  --expected-freeze-sha256 b3295d4cde924720064c04d8c96f3373065e0c6141f00125e62c11dce41c541b \
  --candidates outputs/official-dev-candidates/dev_candidates.json \
  --output-dir outputs/official-dev-final
```

The evaluator verifies the frozen descriptors and source identities, rejects
source-train question/article/exact-context overlap, and predicts without fitting.
All four prediction files are saved before the scorer computes any dev metrics.
The baseline is the first candidate from the same BERT inference pass, using the
shared baseline/candidate selector. The existing train/internal-validation
adapter remains unchanged and still rejects official-dev exports; the new dev
path validates its role explicitly without relabeling data as internal validation.

To reproduce the paired analysis after evaluation:

```bash
python scripts/analyze_official_dev.py --results-dir outputs/official-dev-final
```

This writes a new `paired-analysis.json` and refuses to overwrite an existing one.
The analysis verifies prediction hashes and reproduces the saved scores before
computing the paired comparisons. No models are fitted.

This protocol, implementation, evaluation, and analysis use Codex assistance.
