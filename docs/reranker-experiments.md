# Reranker feature and regularization experiments

Local experiments completed on 11 November 2026 compare answer features, candidate-pool sizes, and ridge regularization on SQuAD v1.1 internal validation. Removing raw span length recovers most of the original reranker's loss. None of the tested feature-rich variants beats BERT's F1. The highest F1, from a logits-only control, is effectively a reproduction of BERT's ranking rather than a demonstrated improvement.

## Scope and method

All trials use the same exported candidates, source splits, scorer, and CPU float64 centered-ridge implementation. BERT is frozen and is not executed or retrained. The full source/split identities, disjoint training/validation memberships, candidate coverage, and first-candidate baseline parity are validated before fitting.

- Training: 12,516 questions from `reranker_train`.
- Evaluation: all 9,131 questions from `internal_val`.
- Training targets: maximum normalized token F1 over training gold annotations, in [0, 1]. Golds do not enter feature extraction.
- Feature means and population standard deviations are fitted on retained training candidates only. Validation does not update weights or preprocessing.
- Candidate order: original BERT score order. Cutoffs retain the first k candidates after validating the unchanged exports.
- Conditions: train 20/evaluate 20; train 20/evaluate 5 with the same frozen fitted model; and train 5/evaluate 5 with a separately fitted model.
- Alpha grid: **0.01, 0.1, 1, 10**. For each feature set and condition, select highest validation F1, then EM, then first configured alpha.

The feature sets and grid were fixed in [the configuration](../configs/reranker-exploration.json) before this sweep, following the earlier length ablation. There are **56 fitted models and 84 evaluations**. Every model has a single fitted state; a model trained on 20 candidates is evaluated on both 20 and 5 candidates without refitting.

The baseline is **65.404 / 79.288 EM/F1**. Results below are percentages; differences are percentage points. Question types use the existing first-word ASCII heuristic (`who`, `what`, `when`, `where`, `why`, `how`, `which`, `other`). Digit and capitalization checks also use the existing ASCII rules.

## Feature definitions

| Profile | Features | Definition |
| --- | ---: | --- |
| `basic_raw_length` | 5 | Start/end logits, raw WordPiece count, digit flag, uppercase flag. |
| `basic_no_length` | 4 | The four basic features excluding WordPiece count. |
| `full_no_length` | 14 | Basic without length plus the existing eight question indicators and two digit/uppercase interactions. |
| `basic_log_length` | 5 | Basic without length plus natural log(1 + WordPiece count). |
| `basic_length_buckets` | 9 | Basic without length plus five indicators for 1, 2–3, 4–7, 8–15, and 16+ WordPieces. |
| `full_type_log_length` | 22 | Full without length plus eight interactions: log(1 + WordPiece count) multiplied by each question type. |
| `logits_only` | 2 | Start and end logits, with all handcrafted features removed. |

The conditional-length profile has a separate log-length slope for each question type and no additional global length column. The eight standalone question indicators are constant within each question and cancel in the centered objective. The interactions can affect rankings because they vary across candidates.

## Results with alpha fixed at one

Each cell is **EM / F1**. These comparisons hold regularization fixed, including the original reference settings.

| Features | Train 20 evaluate 20 | Train 20 evaluate 5 | Train 5 evaluate 5 |
| --- | ---: | ---: | ---: |
| Basic with raw length | 63.925 / 76.782 | 64.221 / 77.365 | 64.155 / 77.300 |
| Basic without length | 65.185 / 79.179 | 65.185 / 79.190 | 65.250 / 79.211 |
| Full without length | 65.250 / 79.196 | 65.250 / 79.199 | 65.261 / 79.196 |
| Basic with log length | 61.965 / 75.004 | 62.786 / 76.305 | 62.162 / 75.811 |
| Basic with length buckets | 51.068 / 72.158 | 52.119 / 72.896 | 60.103 / 76.182 |
| Full with type specific log length | 65.842 / 78.854 | 65.820 / 78.918 | 65.721 / 78.749 |
| Start and end logits only | 65.404 / 79.293 | 65.404 / 79.293 | 65.382 / 79.266 |

The full reranker without length scores 65.250 / 79.196 in the top-20 condition, versus 65.185 / 79.179 for basic without length. That small difference does not establish a useful gain from the existing question-type interactions. With top-5 training, full without length has slightly higher EM but slightly lower F1 than basic without length.

## Results after alpha selection

### Training on twenty and evaluating on twenty

| Features | Selected alpha | EM | F1 |
| --- | ---: | ---: | ---: |
| Basic with raw length | 0.01 | 65.918 | 78.860 |
| Basic without length | 0.1 | 65.272 | 79.256 |
| Full without length | 0.1 | 65.250 | 79.234 |
| Basic with log length | 0.01 | 65.907 | 79.075 |
| Basic with length buckets | 0.01 | 59.260 | 76.237 |
| Full with type specific log length | 0.01 | 65.754 | 79.001 |
| Start and end logits only | 0.01 | 65.404 | 79.293 |

### Training on twenty and evaluating on five

| Features | Selected alpha | EM | F1 |
| --- | ---: | ---: | ---: |
| Basic with raw length | 0.01 | 65.918 | 78.946 |
| Basic without length | 0.1 | 65.272 | 79.257 |
| Full without length | 0.1 | 65.250 | 79.236 |
| Basic with log length | 0.01 | 65.918 | 79.092 |
| Basic with length buckets | 0.01 | 59.336 | 76.341 |
| Full with type specific log length | 0.01 | 65.732 | 79.009 |
| Start and end logits only | 0.01 | 65.404 | 79.293 |

### Training on five and evaluating on five

| Features | Selected alpha | EM | F1 |
| --- | ---: | ---: | ---: |
| Basic with raw length | 0.01 | 65.699 | 78.713 |
| Basic without length | 1 | 65.250 | 79.211 |
| Full without length | 0.1 | 65.228 | 79.199 |
| Basic with log length | 0.01 | 65.579 | 78.545 |
| Basic with length buckets | 0.01 | 64.560 | 78.564 |
| Full with type specific log length | 1 | 65.721 | 78.749 |
| Start and end logits only | 0.01 | 65.393 | 79.277 |

All 84 individual results, including unsuccessful alpha values, are saved in [results.csv](results/internal-validation/results.csv). Alpha is selected separately per feature set and evaluation condition; the tables are exploratory validation selections, not independent test scores.

## Analysis

### Removing length helps more than adding the existing question features

The original alpha-1 top-20 model has a raw length coefficient of approximately -0.00446 per WordPiece. Its start/end coefficients are approximately 0.00964 and 0.00960 per logit. Holding other features fixed, ten additional tokens incur a penalty comparable to losing about 4.6 points of combined BERT logits.

The earlier controlled length ablation changed 1,892 answers, of which 1,890 became longer, and corrected 457 EM errors while introducing 342. The average selected span increased from 2.84 to 4.04 WordPieces. For example, an incomplete answer, “Schubert”, became the complete gold list, “J. S. Bach, Mozart and Schubert”. See [the earlier paired analysis](results/span-length-ablation/comparison.json) and [changed examples](results/span-length-ablation/illustrative-examples.json).

This supports the interpretation that the fitted global length penalty over-favors short answers. A plausible source is correlation between long candidates and poor training targets; the current experiments do not isolate that mechanism. They establish that removing this feature helps this model and dataset, not that answer length is universally harmful.

The existing question-type interactions do not resolve the remaining loss. After alpha selection, basic without length has higher F1 than full without length in every candidate-pool condition.

### Softer length features do not automatically help

At alpha 1, logarithmic length and length buckets perform worse than raw length. The categorical variant is particularly poor with top-20 training. Smaller alpha improves both variants, but neither exceeds BERT F1 anywhere in this grid.

Standardization and the feature basis determine the effective ridge penalty. Replacing raw length with a logarithm or several indicator columns changes both the possible scoring function and its regularization geometry; calling a transform “softer” does not guarantee a weaker learned answer-length preference. A more detailed diagnosis of the bucket model would require inspecting its selected candidates and bucket coefficients.

Question-specific log-length slopes substantially improve on a single global log-length feature at alpha 1, especially with top-20 training. They still do not beat BERT F1, and they do not consistently outperform the globally transformed feature after alpha selection.

### EM and F1 favor different configurations

The highest EM observed is **65.918**, shared by several trials. Among those ties, basic log length trained on 20 and evaluated on 5, with alpha 0.01, has the highest F1: **79.092**. Relative to BERT, that is **+0.515 EM and -0.196 F1**.

That configuration corrects 198 EM errors and breaks 151, a net gain of 47 exact answers. However, its F1 changes decompose as follows:

| Question group | Questions | Contribution to total F1 change |
| --- | ---: | ---: |
| Incorrect to exact | 198 | +1.126 points |
| Exact to incorrect | 151 | -0.796 points |
| Incorrect under both systems | 2,961 | -0.526 points |
| Exact under both systems | 5,821 | 0.000 points |

The reduction in partial credit among answers that remain incorrect outweighs the net F1 benefit from exact-answer corrections. This explains how EM rises while F1 falls. The decomposition is saved in [analysis.json](results/internal-validation/analysis.json).

### The best F1 control is almost unchanged from BERT

The logits-only model trained on 20 achieves **65.403570 EM / 79.292800 F1** for all four alpha values, on both evaluation pools. It changes just three answer texts compared with BERT, corrects no EM errors, and introduces no EM errors. Its F1 advantage is approximately **0.005 points**, not persuasive evidence of a practical gain.

Its two positive coefficients give nearly equal weight to the start and end logits, approximating BERT's original sum. Alpha changes coefficient magnitude without changing the selected answers in these trials. Training this control on five does not improve it: its best F1 is 79.276842.

## Interpretation and next decisions

BERT remains the clearest reference system for balanced EM/F1. Basic without length is a useful ablation demonstrating that the original handcrafted length feature caused much of the loss. The tested question interactions and alternative length encodings do not establish an F1 improvement. If EM is prioritized, retain the smaller-alpha raw/log-length configurations as candidates and explicitly report the accompanying F1 decline.

Internal validation was already used for BERT validation and earlier feature exploration. Searching these 84 evaluations adds selection optimism. These results are exploratory; no significance test or independent generalization gain is claimed. The current sweep does not read official-dev data. Freeze the model, features, alpha, candidate policy, and selection criterion before a final held-out comparison.

## Reproduction and saved artifacts

From the repository root, using the existing project environment and downloaded files:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python scripts/reranker_experiments.py \
  --config configs/reranker-exploration.json \
  --output-dir outputs/reranker-exploration-rerun
```

The output directory must be new. The recorded run used Python 3.10.18, PyTorch 2.8.0, NumPy 2.2.6, CPU float64 ridge, and one Torch thread. The full sweep took approximately 9.4 minutes locally. Timing depends on the machine and its load.

- [Runner](../scripts/reranker_experiments.py) and [configuration](../configs/reranker-exploration.json): reproducible feature definitions and grid.
- [complete.json](results/internal-validation/complete.json): all trials, selected alphas, and baseline scores. Its presence marks successful completion.
- [results.csv](results/internal-validation/results.csv): flat comparison table.
- [manifest.json](results/internal-validation/manifest.json): input/configuration/software hashes and environment.
- Each trial directory: model, evaluation predictions, and metrics. Training targets are saved separately in `targets.json`.
- [verification.json](results/internal-validation/verification.json): all model hashes and exact reload checks, plus six previously recorded alpha-1 reference cases reproduced to within 1e-10.

Small result summaries are versioned under `docs/results/`. Large generated outputs, model weights, and predictions are ignored by Git; retain or share those separately when distributing this report. Hashes identify local artifact bytes and do not authenticate the original BERT training history. The custom feature profiles use the numerical core and do not modify the canonical feature registry or the strict provenance runner. Experiment implementation, execution, and this analysis used Codex assistance.
