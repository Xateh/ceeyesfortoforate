"""Provisional adapter-facing candidate reranker; no gold data at inference."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import torch


def _name(value, description):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string")
    return value


def _number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{description} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"{description} must be finite")
    return value


@dataclass(frozen=True)
class FeatureConfig:
    """Ordered, explicit feature allowlists; names are chosen by the adapter."""
    basic: tuple[str, ...]
    compatibility: tuple[str, ...] = ()
    group: str = "basic"

    def __post_init__(self):
        for field in ("basic", "compatibility"):
            values = getattr(self, field)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"{field} must be a list or tuple")
            object.__setattr__(self, field, tuple(values))
        if self.group not in ("basic", "full") or not self.basic:
            raise ValueError("choose basic/full and provide at least one basic feature")
        names = self.basic + self.compatibility
        for name in names:
            _name(name, "feature name")
            tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
            if tokens & {"gold", "label", "labels", "target", "targets", "f1", "em"}:
                raise ValueError(f"gold-derived feature name is forbidden: {name}")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique across groups")
        if self.group == "full" and not self.compatibility:
            raise ValueError("full requires explicit compatibility features")

    @property
    def names(self):
        return self.basic + (self.compatibility if self.group == "full" else ())

    def to_dict(self):
        return {"basic": list(self.basic), "compatibility": list(self.compatibility),
                "group": self.group}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"basic", "compatibility", "group"}:
            raise ValueError("feature config requires basic, compatibility, group")
        return cls(**value)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    answer: str
    features: Mapping[str, float]


@dataclass(frozen=True)
class Question:
    question_id: str
    candidates: Sequence[Candidate]


def questions_from_records(records):
    """Example local JSON adapter, not a proposed shared pipeline schema."""
    if not isinstance(records, list):
        raise ValueError("questions must be a list")
    questions = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"question_id", "candidates"}:
            raise ValueError("question record requires question_id and candidates only")
        if not isinstance(record["candidates"], list):
            raise ValueError("candidates must be a list")
        candidates = []
        for item in record["candidates"]:
            if not isinstance(item, dict) or set(item) != {"candidate_id", "answer", "features"}:
                raise ValueError("candidate requires candidate_id, answer, features only")
            candidates.append(Candidate(**item))
        questions.append(Question(record["question_id"], candidates))
    return questions


class Reranker:
    """CPU float64, equal-question centered ridge objective, train-only scaling."""

    def __init__(self, features: FeatureConfig, alpha=1.0, no_signal="error"):
        if not isinstance(features, FeatureConfig):
            raise ValueError("features must be FeatureConfig")
        alpha = _number(alpha, "alpha")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if no_signal not in ("error", "first"):
            raise ValueError("no_signal must be error or first")
        self._features = features
        self._alpha = alpha
        self._no_signal = no_signal
        self._state = None

    @property
    def features(self):
        return self._features

    @property
    def alpha(self):
        return self._alpha

    @property
    def no_signal(self):
        return self._no_signal

    def _matrix(self, questions):
        if not isinstance(questions, (list, tuple)) or not questions:
            raise ValueError("provide a nonempty list or tuple of questions")
        rows, groups, seen = [], [], set()
        for question in questions:
            if not isinstance(question, Question):
                raise ValueError("expected Question objects")
            qid = _name(question.question_id, "question_id")
            if qid in seen:
                raise ValueError(f"duplicate question_id: {qid}")
            seen.add(qid)
            if not isinstance(question.candidates, (list, tuple)) or not question.candidates:
                raise ValueError(f"{qid}: candidates must be a nonempty list or tuple")
            start, ids = len(rows), set()
            for candidate in question.candidates:
                if not isinstance(candidate, Candidate):
                    raise ValueError("expected Candidate objects")
                cid = _name(candidate.candidate_id, "candidate_id")
                if cid in ids:
                    raise ValueError(f"{qid}: duplicate candidate_id: {cid}")
                ids.add(cid)
                if not isinstance(candidate.answer, str):
                    raise ValueError(f"{qid}/{cid}: answer must be a string")
                if not isinstance(candidate.features, Mapping):
                    raise ValueError(f"{qid}/{cid}: features must be a mapping")
                row = []
                for feature in self.features.names:
                    if feature not in candidate.features:
                        raise ValueError(f"{qid}/{cid}: missing feature {feature}")
                    row.append(_number(candidate.features[feature], feature))
                rows.append(row)
            groups.append(slice(start, len(rows)))
        return torch.tensor(rows, dtype=torch.float64, device="cpu"), groups

    def fit(self, questions, targets):
        """Targets: {question_id: {candidate_id: overlap_score_in_[0,1]}}.

        Refit replaces state only after successful validation and fitting.
        No validation set or targets are accepted by inference.
        """
        x, groups = self._matrix(questions)
        if not isinstance(targets, Mapping) or set(targets) != {q.question_id for q in questions}:
            raise ValueError("target question IDs must exactly match training questions")
        ys = []
        informative = []
        for question, group in zip(questions, groups):
            labels = targets[question.question_id]
            if not isinstance(labels, Mapping) or set(labels) != {c.candidate_id for c in question.candidates}:
                raise ValueError(f"{question.question_id}: target candidate IDs must match exactly")
            y = [_number(labels[c.candidate_id], "target") for c in question.candidates]
            if any(value < 0 or value > 1 for value in y):
                raise ValueError("overlap targets must be in [0, 1]")
            ys.extend(y)
            if max(y) > min(y):
                informative.append(group)
        if not informative and self.no_signal == "error":
            raise ValueError("no within-question target signal; use no_signal='first' explicitly")
        mean = x.mean(dim=0)
        scale = x.std(dim=0, correction=0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        z = (x - mean) / scale
        y = torch.tensor(ys, dtype=torch.float64, device="cpu")
        width = len(self.features.names)
        gram = torch.zeros((width, width), dtype=torch.float64, device="cpu")
        rhs = torch.zeros(width, dtype=torch.float64, device="cpu")
        # Each informative question has equal total weight, irrespective of count.
        for group in informative:
            a = z[group] - z[group].mean(dim=0)
            b = y[group] - y[group].mean()
            weight = 1.0 / (len(informative) * len(b))
            gram += weight * (a.T @ a)
            rhs += weight * (a.T @ b)
        coef = torch.linalg.solve(gram + self.alpha * torch.eye(width, dtype=torch.float64, device="cpu"), rhs)
        if not all(torch.isfinite(t).all().item() for t in (mean, scale, z, coef)):
            raise ValueError("numeric overflow during fitting; rescale inputs")
        self._state = {"mean": mean, "scale": scale, "coef": coef,
                       "training_questions": len(questions),
                       "informative_questions": len(informative)}
        return self

    def scores(self, questions):
        if self._state is None:
            raise ValueError("model is not fitted")
        x, groups = self._matrix(questions)
        state = self._state
        scores = ((x - state["mean"]) / state["scale"]) @ state["coef"]
        if not torch.isfinite(scores).all().item():
            raise ValueError("numeric overflow during prediction; rescale inputs")
        return [scores[group].tolist() for group in groups]

    def select(self, questions):
        """Select exactly one supplied candidate; first input candidate wins ties."""
        scores = self.scores(questions)
        return {q.question_id: q.candidates[max(range(len(s)), key=s.__getitem__)]
                for q, s in zip(questions, scores)}

    def predict(self, questions):
        return {qid: candidate.answer for qid, candidate in self.select(questions).items()}

    def save_predictions(self, questions, path):
        _write_json(path, self.predict(questions))

    def save(self, path):
        if self._state is None:
            raise ValueError("model is not fitted")
        state = {key: value.tolist() if isinstance(value, torch.Tensor) else value
                 for key, value in self._state.items()}
        _write_json(path, {"version": 1, "objective": "question_centered_ridge",
                          "features": self.features.to_dict(), "alpha": self.alpha,
                          "no_signal": self.no_signal, "state": state})

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {"version", "objective", "features", "alpha", "no_signal", "state"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("invalid model artifact fields")
        if type(data["version"]) is not int or data["version"] != 1 or data["objective"] != "question_centered_ridge":
            raise ValueError("unsupported model format or objective")
        model = cls(FeatureConfig.from_dict(data["features"]), data["alpha"], data["no_signal"])
        state = data["state"]
        if not isinstance(state, dict) or set(state) != {"mean", "scale", "coef", "training_questions", "informative_questions"}:
            raise ValueError("invalid preprocessing/model state")
        validated = {}
        for key in ("mean", "scale", "coef"):
            value = state[key]
            if not isinstance(value, list) or len(value) != len(model.features.names):
                raise ValueError(f"invalid {key} dimensions")
            values = [_number(v, key) for v in value]
            if key == "scale" and any(v <= 0 for v in values):
                raise ValueError("scale must be positive")
            validated[key] = torch.tensor(values, dtype=torch.float64, device="cpu")
        for key in ("training_questions", "informative_questions"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid {key}")
            validated[key] = state[key]
        count, informative = validated["training_questions"], validated["informative_questions"]
        if count < 1 or informative > count:
            raise ValueError("invalid training counts")
        if informative == 0 and (model.no_signal != "first" or torch.count_nonzero(validated["coef"]).item()):
            raise ValueError("invalid no-signal model")
        model._state = validated
        return model


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
