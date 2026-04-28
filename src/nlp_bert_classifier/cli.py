from __future__ import annotations

import argparse
import json

from .pipeline import TrainConfig, predict_text, train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bert-text-classifier",
        description="CLI для донавчання BERT на задачі класифікації тексту.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Запустити донавчання моделі.")
    train_parser.add_argument("--model-name", default=TrainConfig.model_name)
    train_parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    train_parser.add_argument("--train-samples", type=int, default=TrainConfig.train_samples)
    train_parser.add_argument("--test-samples", type=int, default=TrainConfig.test_samples)
    train_parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    train_parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    train_parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Класифікувати довільний текст.",
    )
    predict_parser.add_argument("--model-dir", default=f"{TrainConfig.output_dir}/model")
    predict_parser.add_argument("--text", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = TrainConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        metrics = train_and_evaluate(config)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    if args.command == "predict":
        result = predict_text(args.model_dir, args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    parser.error(f"Невідома команда: {args.command}")


if __name__ == "__main__":
    main()
