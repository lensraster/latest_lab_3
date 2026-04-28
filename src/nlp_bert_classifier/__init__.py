"""Навчальний пакет для донавчання BERT."""

from .pipeline import (
    build_compute_metrics,
    load_and_prepare_ag_news,
    predict_text,
    train_and_evaluate,
)

__all__ = [
    "build_compute_metrics",
    "load_and_prepare_ag_news",
    "predict_text",
    "train_and_evaluate",
]
