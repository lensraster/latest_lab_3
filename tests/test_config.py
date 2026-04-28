import pytest

from nlp_bert_classifier.pipeline import TrainConfig


def test_train_config_validation_passes() -> None:
    config = TrainConfig(train_samples=16, test_samples=8, epochs=1, batch_size=2)
    config.validate()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("train_samples", 0),
        ("test_samples", 0),
        ("epochs", 0),
        ("batch_size", 0),
        ("max_length", 0),
    ],
)
def test_train_config_validation_rejects_non_positive_values(field_name: str, value: int) -> None:
    kwargs = {
        "train_samples": 8,
        "test_samples": 8,
        "epochs": 1,
        "batch_size": 2,
        "max_length": 64,
    }
    kwargs[field_name] = value
    config = TrainConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()
