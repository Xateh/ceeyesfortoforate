# Data splits

`scripts/split_data.py` divides the SQuAD v1.1 training set into three: `bert_train`, `reranker_train`, and `internal_val`. 

Split fractions and seed are taken from `configs/split.json` (seed 4248, fractions 0.75 / 0.15 / 0.10). Articles are separate across splits.

## Split sizes (seed 4248)

| Split | Articles | Paragraphs | Questions | % of training questions |
| --- | ---: | ---: | ---: | ---: |
| `bert_train` | 332 | 14,220 | 65,952 | 75.3% |
| `reranker_train` | 66 | 2,718 | 12,516 | 14.3% |
| `internal_val` | 44 | 1,958 | 9,131 | 10.4% |
| **Training total** | 442 | 18,896 | 87,599 | 100% |
| `dev-v1.1` (untouched) | 48 | 2,067 | 10,570 | — |

## Running

Download the data from Canvas (Project -> qa), then place in the `data` folder and run:

```bash
python scripts/split_data.py           # writes data/splits/
python scripts/split_data.py --verify  # recomputes and checks the saved files
```