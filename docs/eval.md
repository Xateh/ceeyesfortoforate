# Evaluation

`squad_eval.py` contains `evaluate(dataset, predictions)`, to be imported and used for evaluating models.
- `dataset`: path to the json file containing original validation data (in text)
- `predictions`: a dictionary of `{question id: model prediction}`

Example usage:
```python
from squad_eval import evaluate
metrics = evaluate("data/splits/internal_val.json", predictions) 
metrics["exact"], metrics["f1"], metrics["total"]
```
Place the official `evaluate-v2.0.py` in the root folder.

## BERT baseline evaluation

```bash
python scripts/bert_predict.py --checkpoint --split --max-answer-length --output-dir --batch-size
```

For each question, over all its windows:

- consider the top `n_best` (20) start and end positions of each window;
- a span is valid if both ends are context tokens (not the question, `[CLS]`,
  `[SEP]` or padding), start ≤ end, and it is at most `max_answer_length` (30)
  WordPiece tokens;
- score = start logit + end logit; the highest-scoring valid span wins;
- overlapping windows all compete using raw scores; no max-context filtering is applied.

## Baseline results

| Split | EM | F1 | Checkpoint | Settings |
| --- | ---: | ---: | --- | --- |
| `internal_val` | 65.40% | 79.29% | `bert-base-baseline/final` | max answer length 30, n-best 20, all windows |
| `dev` | 80.02% | 87.55% | `bert-base-baseline/final` | max answer length 30, n-best 20, all windows |

See [candidate generation](candidates.md) for label-free top-20 span export.
