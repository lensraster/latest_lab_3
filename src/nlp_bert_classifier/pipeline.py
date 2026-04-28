from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_DATASET_NAME = "ag_news"


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_name: str = DEFAULT_DATASET_NAME
    output_dir: str = "artifacts/bert-ag-news"
    max_length: int = 128
    train_samples: int = 1200
    test_samples: int = 400
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42

    def validate(self) -> None:
        if self.train_samples <= 0:
            raise ValueError("train_samples має бути > 0")
        if self.test_samples <= 0:
            raise ValueError("test_samples має бути > 0")
        if self.epochs <= 0:
            raise ValueError("epochs має бути > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size має бути > 0")
        if self.max_length <= 0:
            raise ValueError("max_length має бути > 0")


def build_compute_metrics() -> Callable:
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy_result = accuracy.compute(predictions=predictions, references=labels)
        f1_result = f1.compute(
            predictions=predictions,
            references=labels,
            average="weighted",
        )
        return {
            "accuracy": float(accuracy_result["accuracy"]),
            "f1_weighted": float(f1_result["f1"]),
        }

    return compute_metrics


def load_and_prepare_ag_news(
    tokenizer_name: str,
    max_length: int = 128,
    train_samples: int = 1200,
    test_samples: int = 400,
) -> tuple[DatasetDict, dict[int, str]]:
    dataset = load_dataset(DEFAULT_DATASET_NAME)

    available_train = len(dataset["train"])
    available_test = len(dataset["test"])
    train_size = min(train_samples, available_train)
    test_size = min(test_samples, available_test)

    dataset = DatasetDict(
        {
            "train": dataset["train"].shuffle(seed=42).select(range(train_size)),
            "test": dataset["test"].shuffle(seed=42).select(range(test_size)),
        }
    )

    label_names = dataset["train"].features["label"].names
    id_to_label = {idx: name for idx, name in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized, id_to_label


def train_and_evaluate(config: TrainConfig) -> dict[str, float]:
    config.validate()
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenized_dataset, id_to_label = load_and_prepare_ag_news(
        tokenizer_name=config.model_name,
        max_length=config.max_length,
        train_samples=config.train_samples,
        test_samples=config.test_samples,
    )
    label_to_id = {label: idx for idx, label in id_to_label.items()}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()

    export_dir = output_path / "model"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    (output_path / "metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "metrics": {k: float(v) for k, v in metrics.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {k: float(v) for k, v in metrics.items()}


def predict_text(model_dir: str, text: str) -> dict[str, float | str]:
    if not text.strip():
        raise ValueError("Текст для класифікації не може бути порожнім.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    inputs = tokenizer(text, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    label = model.config.id2label.get(pred_idx, str(pred_idx))

    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
    }
