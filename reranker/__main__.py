"""Small local JSON adapter. Run python -m reranker --help."""
import argparse
import json
from pathlib import Path

from .core import FeatureConfig, Reranker, questions_from_records


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--questions", required=True)
    train.add_argument("--targets", required=True)
    train.add_argument("--features", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--no-signal", choices=("error", "first"), default="error")
    predict = commands.add_parser("predict")
    predict.add_argument("--questions", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        questions = questions_from_records(read(args.questions))
        if args.command == "train":
            model = Reranker(FeatureConfig.from_dict(read(args.features)), args.alpha, args.no_signal)
            model.fit(questions, read(args.targets)).save(args.model)
        else:
            Reranker.load(args.model).save_predictions(questions, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"reranker: {error}\n")


if __name__ == "__main__":
    main()
