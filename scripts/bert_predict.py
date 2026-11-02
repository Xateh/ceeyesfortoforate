"""Baseline: answer internal_val with the fine-tuned BERT checkpoint and score with squad_eval.

For each question, every window is scored; the answer is the best valid span
over all of the question's windows (so overlapping windows compete):
    score = start_logit + end_logit
    valid = start and end are context tokens, start <= end, at most
            --max-answer-length tokens long (top 20 starts/ends per window).

Needs data/tokenized/internal_val from scripts/qa_preprocessing.py.

Usage:
    python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_examples
from squad_eval import evaluate

EXAMPLES = "data/splits/internal_val.json"
FEATURES = "data/tokenized/internal_val"
N_BEST = 20  # top start/end positions considered per window (Google's n_best_size)

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/bert-base-baseline/final")
    parser.add_argument("--max-answer-length", type=int, default=30,
                        help="Longest answer in tokens (Google's default: 30).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert-base-baseline"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint).to(device).eval()
    features = load_from_disk(FEATURES)

    # start/end scores for every window.
    inputs = features.select_columns(["input_ids", "token_type_ids", "attention_mask"])
    start_logits, end_logits = [], []
    with torch.no_grad():
        for batch in inputs.with_format("torch").iter(batch_size=args.batch_size):
            output = model(**{key: value.long().to(device) for key, value in batch.items()})
            start_logits.append(output.start_logits.cpu().numpy())
            end_logits.append(output.end_logits.cpu().numpy())
    start_logits, end_logits = np.concatenate(start_logits), np.concatenate(end_logits)

    # best valid span per question across all windows
    contexts = {example["id"]: example["context"] for example in load_examples(EXAMPLES)}
    best = {}  # question id -> (score, answer text)
    for i, (qid, offsets) in enumerate(zip(features["example_id"], features["offset_mapping"])):
        for s in np.argsort(start_logits[i])[-N_BEST:]:
            for e in np.argsort(end_logits[i])[-N_BEST:]:
                if offsets[s] is None or offsets[e] is None:  # question, special or padding token
                    continue
                if e < s or e - s + 1 > args.max_answer_length:
                    continue
                score = start_logits[i][s] + end_logits[i][e]
                if qid not in best or score > best[qid][0]:
                    best[qid] = (score, contexts[qid][offsets[s][0]:offsets[e][1]])
    predictions = {qid: best[qid][1] if qid in best else "" for qid in contexts}

    # score & save
    scores = evaluate(EXAMPLES, predictions)
    metrics = {"exact": scores["exact"], "f1": scores["f1"], "total": scores["total"],
               "checkpoint": str(args.checkpoint), "max_answer_length": args.max_answer_length,
               "n_best": N_BEST}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "internal_val_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "internal_val_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"internal_val: EM {metrics['exact']:.2f}, F1 {metrics['f1']:.2f} "
          f"({metrics['total']} questions); saved to {args.output_dir}")


if __name__ == "__main__":
    main()