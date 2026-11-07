"""Shared baseline/candidate postprocessing; no model or gold labels required."""

import hashlib
import json
from collections import defaultdict

import numpy as np

N_BEST = 20


def select_candidates(contexts, example_ids, offset_mappings, start_logits,
                      end_logits, max_answer_length=30, max_candidates=20):
    """Return ranked unique spans per question, preserving baseline tie order.

    Logit addition uses the input NumPy dtype, exactly as the original baseline.
    All window indices refer to rows in the tokenized split.
    """
    if max_answer_length < 1 or max_candidates < 1:
        raise ValueError("Answer length and candidate count must be positive")
    if not (len(example_ids) == len(offset_mappings) == len(start_logits) == len(end_logits)):
        raise ValueError("Feature and logit row counts differ")
    windows = defaultdict(list)
    for i, qid in enumerate(example_ids):
        if qid not in contexts:
            raise ValueError(f"Unknown question ID in features: {qid}")
        windows[qid].append(i)

    result = {}
    for qid, context in contexts.items():
        unique = {}
        order = 0
        for i in windows[qid]:
            offsets = offset_mappings[i]
            if len(offsets) != len(start_logits[i]) or len(offsets) != len(end_logits[i]):
                raise ValueError(f"{qid}: offset/logit length mismatch in window {i}")
            if not (np.isfinite(start_logits[i]).all() and np.isfinite(end_logits[i]).all()):
                raise ValueError(f"{qid}: nonfinite logits in window {i}")
            # Keep the original baseline's ascending traversal of the top positions.
            for s in np.argsort(start_logits[i])[-N_BEST:]:
                for e in np.argsort(end_logits[i])[-N_BEST:]:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    start, end = int(offsets[s][0]), int(offsets[e][1])
                    if not 0 <= start < end <= len(context):
                        raise ValueError(f"{qid}: invalid character offsets in window {i}")
                    score = float(start_logits[i][s] + end_logits[i][e])
                    if not np.isfinite(score):
                        raise ValueError(f"{qid}: nonfinite combined score in window {i}")
                    key = (start, end)
                    candidate = {
                        "answer": context[start:end], "start": start, "end": end,
                        "window_index": i, "start_token_index": int(s),
                        "end_token_index": int(e), "span_token_count": int(e - s + 1),
                        "start_logit": float(start_logits[i][s]),
                        "end_logit": float(end_logits[i][e]), "combined_score": score,
                    }
                    if key not in unique or score > unique[key][0]["combined_score"]:
                        unique[key] = (candidate, order)
                    order += 1
        ranked = sorted(unique.values(), key=lambda item: (-item[0]["combined_score"], item[1]))
        result[qid] = [candidate for candidate, _ in ranked[:max_candidates]]
    return result


def candidate_records(examples, candidates):
    """Build a label-free export; refuse questions without real candidates."""
    missing = [example["id"] for example in examples if not candidates.get(example["id"])]
    if missing:
        raise ValueError(f"No valid candidates for {len(missing)} question(s): {missing[:10]}")
    records = []
    for example in examples:
        # SQuAD article titles identify articles; include title in paragraph identity.
        identity = json.dumps([example["title"], example["context"]], ensure_ascii=False)
        records.append({
            "question_id": example["id"], "question": example["question"],
            "article_id": example["title"],
            "context_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "context": example["context"], "candidates": candidates[example["id"]],
        })
    return records
