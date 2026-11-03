"""Make baseline predictions or export top-20 spans using a BERT checkpoint.

For each question, every window is scored; the answer is the best valid span
over all of the question's windows (so overlapping windows compete):
    score = start_logit + end_logit
    valid = start and end are context tokens, start <= end, at most
            --max-answer-length tokens long (top 20 starts/ends per window).

Usage:
    python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split dev
    python scripts/bert_predict.py --split reranker_train --candidates
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from span_candidates import N_BEST, candidate_records, select_candidates

EXAMPLES = {
    "reranker_train": "data/splits/reranker_train.json",
    "internal_val": "data/splits/internal_val.json",
    "dev": "data/dev-v1.1.json",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/bert-base-baseline/final")
    parser.add_argument("--split", choices=EXAMPLES, default="internal_val",
                        help="reranker_train, internal_val or dev (only for final evaluation)")
    parser.add_argument("--max-answer-length", type=int, default=30,
                        help="Longest answer in tokens (Google's default: 30).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert-base-baseline"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", action="store_true",
                        help="Save up to 20 unique spans per question without evaluating gold answers.")
    args = parser.parse_args()
    if args.max_answer_length < 1 or args.batch_size < 1:
        parser.error("--max-answer-length and --batch-size must be positive")
    if args.split == "reranker_train" and not args.candidates:
        parser.error("reranker_train requires --candidates")
    examples_path, features_path = EXAMPLES[args.split], f"data/tokenized/{args.split}"

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForQuestionAnswering
    from preprocessing import load_examples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoint).to(device).eval()
    features = load_from_disk(features_path)

    if len(features) == 0:
        raise ValueError(f"No tokenized windows found for {args.split}")

    # start/end scores for every window.
    inputs = features.select_columns(["input_ids", "token_type_ids", "attention_mask"])
    start_logits, end_logits = [], []
    with torch.no_grad():
        for batch in inputs.with_format("torch").iter(batch_size=args.batch_size):
            output = model(**{key: value.long().to(device) for key, value in batch.items()})
            start_logits.append(output.start_logits.cpu().numpy())
            end_logits.append(output.end_logits.cpu().numpy())
    start_logits, end_logits = np.concatenate(start_logits), np.concatenate(end_logits)

    examples = load_examples(examples_path)
    contexts = {example["id"]: example["context"] for example in examples}
    if len(contexts) != len(examples):
        raise ValueError("Duplicate question IDs in source data")
    candidates = select_candidates(
        contexts, features["example_id"], features["offset_mapping"],
        start_logits, end_logits, args.max_answer_length,
        max_candidates=20 if args.candidates else 1,
    )
    predictions = {qid: spans[0]["answer"] if spans else "" for qid, spans in candidates.items()}

    if args.candidates:
        payload = {
            "schema_version": 1,
            "metadata": {
                "split": args.split, "checkpoint": str(args.checkpoint),
                "examples_path": examples_path, "features_path": features_path,
                "max_answer_length": args.max_answer_length, "n_best": N_BEST,
                "max_candidates": 20, "batch_size": args.batch_size,
                "device": str(device), "numpy_version": np.__version__,
                "torch_version": torch.__version__, "logit_dtype": str(start_logits.dtype),
                "tie_break": "first occurrence in baseline traversal",
                "max_context_only": False,
            },
            "questions": candidate_records(examples, candidates),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.split}_candidates.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
        print(f"{args.split}: saved candidates for {len(examples)} questions to {path}")
        return

    # score & save
    from squad_eval import evaluate
    scores = evaluate(examples_path, predictions)
    metrics = {"split": args.split, "exact": scores["exact"], "f1": scores["f1"], "total": scores["total"],
               "checkpoint": str(args.checkpoint), "max_answer_length": args.max_answer_length,
               "n_best": N_BEST}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"{args.split}: EM {metrics['exact']:.2f}, F1 {metrics['f1']:.2f} "
          f"({metrics['total']} questions); saved to {args.output_dir}")


if __name__ == "__main__":
    main()
