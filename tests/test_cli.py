from nlp_bert_classifier.cli import build_parser


def test_cli_parser_has_required_subcommands() -> None:
    parser = build_parser()
    train_args = parser.parse_args(["train"])
    assert train_args.command == "train"

    predict_args = parser.parse_args(["predict", "--text", "Hello"])
    assert predict_args.command == "predict"
    assert predict_args.text == "Hello"
