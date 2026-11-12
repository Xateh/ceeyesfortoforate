"""Shared SQuAD evaluation for all systems

Usage:
    from squad_eval import evaluate
    metrics = evaluate("data/splits/internal_val.json", predictions)  # {question_id: answer}
    metrics["exact"], metrics["f1"], metrics["total"]
"""

import importlib.util
import json
from pathlib import Path

EVALUATOR_PATH = Path(__file__).resolve().parent.parent / "evaluate-v2.0.py"

_spec = importlib.util.spec_from_file_location("official_squad_evaluator", EVALUATOR_PATH)
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def evaluate(dataset, predictions):
    if isinstance(dataset, (str, Path)):
        with open(dataset, encoding="utf-8") as handle:
            dataset = json.load(handle)["data"]
    question_ids = {qa["id"] for article in dataset
                    for paragraph in article["paragraphs"] for qa in paragraph["qas"]}
    missing, unknown = question_ids - set(predictions), set(predictions) - question_ids
    if missing or unknown:
        raise ValueError(f"Predictions must cover exactly the dataset's questions: "
                         f"{len(missing)} missing, {len(unknown)} not in the dataset.")
    exact_raw, f1_raw = official.get_raw_scores(dataset, predictions)
    metrics = dict(official.make_eval_dict(exact_raw, f1_raw))
    metrics.update(exact_per_question=exact_raw, f1_per_question=f1_raw)
    return metrics