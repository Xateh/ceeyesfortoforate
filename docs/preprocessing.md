# Preprocessing

`scripts/qa_preprocessing.py` turns SQuAD examples into BERT inputs and save them to `data/tokenized`. It also checks every split and prints representative examples.

Settings are in `configs/bert_qa.json`: `google-bert/bert-base-uncased` at the
pinned revision, `max_length` 384, `doc_stride` 128. 384 is the standard BERT
SQuAD setting. 98–99% of questions fit in one window.

## What a feature contains

Each question + context becomes one or more features (windows):

```
[CLS] question [SEP] context window [SEP] [PAD] ...   (384 tokens)
```

| Column | Meaning |
| --- | --- |
| `input_ids`, `token_type_ids`, `attention_mask` | BERT inputs |
| `start_positions`, `end_positions` | Answer token labels; both 0 (`[CLS]`) if the answer is not fully inside this window |
| `example_id` | SQuAD question ID, to group a question's windows at prediction time |
| `offset_mapping` | `(char_start, char_end)` in the original context for context tokens; `None` for question, special and padding tokens |

## Token handling

- Contexts that do not fit are split into windows sharing
  exactly 128 context tokens.
- Windows are the answers are absent / cut-off are labelled `(0, 0)` (`[CLS]`).

## Running

```bash
python scripts/preprocessing.py
```

For each split it prints the counts below, then one example question for each
type. Checks are performed to ensure at least 1 window contains the full answer and the labels are correct. 

## Usage

To use the tokenized data in a training script:

```python
from datasets import load_from_disk
train = load_from_disk("data/tokenized/bert_train").remove_columns(["example_id", "offset_mapping"])
```

## Expected Results

| Count | `bert_train` | `reranker_train` | `internal_val` | `dev` |
| --- | ---: | ---: | ---: | ---: |
| questions | 65,952 | 12,516 | 9,131 | 10,570 |
| windows (features) | 66,606 | 12,664 | 9,254 | 10,784 |
| answer contained in 1 window, exact match | 65,899 | 12,528 | 9,132 | 10,580 |
| gold ends mid-token | 183 | 19 | 24 | 32 |
| answer absent from window | 520 | 117 | 97 | 168 |
| windows: answer cut-off in window (truncated) | 4 | 0 | 1 | 4 |


