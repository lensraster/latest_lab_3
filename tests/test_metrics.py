import numpy as np

from nlp_bert_classifier.pipeline import build_compute_metrics


def test_compute_metrics_returns_expected_keys() -> None:
    compute_metrics = build_compute_metrics()
    logits = np.array(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    labels = np.array([1, 0, 2])
    metrics = compute_metrics((logits, labels))
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert metrics["accuracy"] == 1.0
