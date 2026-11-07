# Candidate generation (issue #7)

Run from the repository root after preprocessing and BERT training:

```bash
python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split reranker_train --candidates
python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split internal_val --candidates
# Reserve official dev for the final evaluation:
python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split dev --candidates
```

Outputs go to `outputs/bert-base-baseline/<split>_candidates.json`; change the
folder with `--output-dir`. `--max-answer-length` defaults to 30 and `--batch-size`
to 64. Candidate mode does not invoke evaluation or export gold answers, labels,
or features. Normal baseline prediction/evaluation remains the default for
`internal_val` and `dev`; `reranker_train` requires `--candidates`.

## Selection policy

Both modes share `scripts/span_candidates.py`. For every window, consider the
same top 20 start and top 20 end positions as the original baseline. Reject
non-context endpoints, reversed spans, and spans longer than the token limit.
All windows compete; no max-context filtering or score normalization is applied.
This is the best 20 within that search, not an exhaustive search of all pairs.

Deduplicate by `(start, end)` within each question's context, retaining the
highest-scoring occurrence and its complete start/end/combined score triplet.
Identical answer strings at different offsets remain separate. Sort by combined
score descending, breaking ties by first encounter of the winning occurrence:
feature rows in dataset order, then the original `np.argsort(...)[-20:]` start
and end loops. Equal-scoring duplicate occurrences keep their first window.
Retain up to 20 unique spans; fewer are allowed. Candidate 1 therefore matches
the baseline for the same inputs and settings. No-candidate questions fail the
export with question IDs before any output file is written; normal baseline mode
continues to emit an empty answer for those questions.

## JSON format (schema_version 1)

Top-level fields are `schema_version`, `metadata`, and `questions`. Metadata
records split, checkpoint path, input paths, answer/search/candidate limits,
batch size, device, NumPy/PyTorch versions, logit dtype, and tie/window policies.
Each question has:

| Field | Meaning |
| --- | --- |
| `question_id`, `question` | Original SQuAD ID and preprocessing's stripped question |
| `article_id` | Original SQuAD article title |
| `context_id` | SHA-256 of UTF-8 `json.dumps([title, context], ensure_ascii=False)` |
| `context` | Exact original paragraph; no normalization |
| `candidates` | Ordered list, best candidate first |

Each candidate has:

| Field | Meaning |
| --- | --- |
| `answer` | Exact `context[start:end]` |
| `start`, `end` | Zero-based, half-open Unicode code-point offsets, not bytes or UTF-16 |
| `window_index` | Zero-based row in the tokenized split (global, not question-local) |
| `start_token_index`, `end_token_index` | Inclusive token positions in that window |
| `span_token_count` | `end_token_index - start_token_index + 1` |
| `start_logit`, `end_logit` | Raw logits from the retained occurrence |
| `combined_score` | Their sum in the model's NumPy logit dtype, as in baseline |

Float32 combined scores can differ slightly from adding the exported logits as
Python float64 values. Do not change their precision to reorder candidates.
The provisional PR #20 bundle has different schema/tie rules and requires an
adapter; this file is not a validated reranker bundle. Features and targets are
left to #8/#9.

## Verification and reproducibility

```bash
python -m unittest discover -s tests -p 'test_span_candidates.py' -v
```

Offline fixtures cover window deduplication, ties, filters, shortlist sizes,
Unicode slices, label-free serialization, and comparison with the original
baseline loop on random and tied float32/float64 logits. They do not establish
real checkpoint accuracy. For a real-data check, run baseline prediction and
candidate generation on the same internal-validation inputs and verify each
first candidate's `answer` equals `<split>_predictions.json[question_id]`.

Keep the checkpoint, source split, tokenized dataset, software versions and
hardware fixed for reproducible runs. Metadata paths identify inputs but are
not content hashes; replacing files at those paths invalidates the comparison.
Inference is in eval mode; bitwise equality across devices is not guaranteed.
