"""Lightweight candidate scoring with explicit adapter boundaries."""
from .core import Candidate, FeatureConfig, Question, Reranker, questions_from_records

__all__ = ["Candidate", "FeatureConfig", "Question", "Reranker", "questions_from_records"]
