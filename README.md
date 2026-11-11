# CS4248 G19 — BERT QA answer reranking

We compare a fine-tuned BERT extractive QA baseline with basic and question-type-aware
linear rerankers on SQuAD v1.1. The rerankers choose among supplied BERT spans;
they do not generate new answers or retrain BERT.

## Setup

Use Python 3.11 or 3.12 and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
# CPU build; sufficient for reranker training and the offline tests.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. On macOS, install
`torch==2.8.0` without the CPU index. For BERT GPU work, install a compatible
[PyTorch 2.8 CUDA build](https://pytorch.org/get-started/previous-versions/#v280)
instead of the CPU build; RTX 50-series requires the CUDA 12.8 option.
Use an allocated compute node for cluster workloads, not a login node.

An offline environment check uses a tiny randomly initialized model:

```bash
python scripts/smoke_test.py --tiny --device cpu --output outputs/smoke-cpu.json
```

This checks software, not QA accuracy. See [setup validation](docs/setup-validation.md)
for recorded environment checks. Keep data, weights and generated outputs outside Git.

## Pipeline

| Step | Entry point / guide |
| --- | --- |
| Article-disjoint train-derived splits | [Data splits](docs/data-split.md) |
| BERT tokenization | [Preprocessing](docs/preprocessing.md) |
| BERT training and baseline predictions | `scripts/bert_train.py`, `scripts/bert_predict.py` |
| Candidate adaptation, reranker training and reload | [Reranker guide](docs/reranker.md) |
| Shared EM/F1 evaluation | [Evaluation](docs/eval.md) |

The team split uses seed 4248 and 75/15/10 article allocation. BERT optimizer
seed 42 is separate. Reranker fitting uses only reranker_train; official dev is
reserved for frozen final comparisons. Team documentation already reports baseline
dev scores, so dev must not be described as untouched.

## Current reranker result

Both variants have completed provisional real-data fits and exact save/reload
checks. On 9,131 internal-validation questions, supplied BERT scored **65.40 EM /
79.29 F1**, basic **63.93 / 76.78**, and full **63.89 / 76.77**. Neither reranker
beat the baseline. The [guide](docs/reranker.md) explains the policy, inputs,
commands and unresolved provenance; internal validation was also used upstream.

Additional [feature and regularization experiments](docs/reranker-experiments.md)
record 56 local fits and 84 exploratory internal-validation comparisons, including
full features without span length, alternative length encodings, and an alpha grid.

The [frozen official-dev comparison](docs/official-dev-evaluation.md) records the
final BERT/basic/full comparison and a matched-alpha length ablation on all
10,570 dev questions. The two main selected rerankers did not improve BERT's EM or F1.

Xu An's reranker work builds on Hai's data/BERT/evaluator pipeline and Sean's
candidate exporter. Implementation, tests, review and documentation used Codex
assistance. Each member should review and disclose their actual AI use in the report.
