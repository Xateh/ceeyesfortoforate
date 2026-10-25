"""Split SQuAD v1.1 training articles into BERT-training, reranker-training and
internal-validation sets, by article (title).

Usage:
    python scripts/split_data.py                 # write the three split files
    python scripts/split_data.py --verify        # recompute and compare with saved splits
"""

import argparse
import json
import random
import sys
from pathlib import Path

SPLITS = ("bert_train", "reranker_train", "internal_val")


def load_squad(path):
    with open(path, encoding="utf-8") as handle:
        squad = json.load(handle)
    return squad

def question_ids(articles):
    return [qa["id"] for article in articles
            for paragraph in article["paragraphs"] for qa in paragraph["qas"]]

def stats(articles):
    return {
        "articles": len(articles),
        "paragraphs": sum(len(article["paragraphs"]) for article in articles),
        "questions": len(question_ids(articles)),
    }

def assign_articles(titles, fractions, seed):
    order = sorted(titles)
    random.Random(seed).shuffle(order)
    n_total = len(order)
    n_val = round(n_total * fractions["internal_val"])
    n_reranker = round(n_total * fractions["reranker_train"])
    if n_val < 1 or n_reranker < 1 or n_val + n_reranker >= n_total:
        raise ValueError("Fractions leave an empty split.")
    assignment = {}
    for index, title in enumerate(order):
        if index < n_val:
            assignment[title] = "internal_val"
        elif index < n_val + n_reranker:
            assignment[title] = "reranker_train"
        else:
            assignment[title] = "bert_train"
    return assignment

def build_splits(train, fractions, seed):
    articles = train["data"]
    titles = [article["title"] for article in articles]
    assignment = assign_articles(titles, fractions, seed)
    split_articles = {name: [a for a in articles if assignment[a["title"]] == name]
                      for name in SPLITS}
    return assignment, split_articles

def write_squad(path, articles):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"data": articles, "version": "1.1"}, handle, ensure_ascii=False)

def contents(articles):
    return {a["title"] for a in articles}, set(question_ids(articles))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/split.json"))
    parser.add_argument("--train", type=Path, default=Path("data/train-v1.1.json"))
    parser.add_argument("--dev", type=Path, default=Path("data/dev-v1.1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--verify", action="store_true",
                        help="Recompute the splits and check they match the saved files.")
    args = parser.parse_args()

    # load configs
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed, fractions = config["seed"], config["fractions"]
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-9:
        parser.error(f"Config fractions must cover {SPLITS} and sum to 1.")

    train, dev = load_squad(args.train), load_squad(args.dev)
    _, split_articles = build_splits(train, fractions, seed)

    # check reproducibility
    if args.verify:
        problems = []
        saved = {}
        for name in SPLITS:
            path = args.output_dir / f"{name}.json"
            if not path.exists():
                problems.append(f"missing {path}")
                continue
            saved[name] = load_squad(path)["data"]
            if contents(saved[name]) != contents(split_articles[name]):
                problems.append(f"{path}: articles or question IDs differ from a fresh computation")
            elif saved[name] != split_articles[name]:
                problems.append(f"{path}: same IDs but question/answer content differs")
                
        if problems:
            print("FAIL:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("PASS: saved splits reproduced correctly.")
        
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in SPLITS:
            write_squad(args.output_dir / f"{name}.json", split_articles[name])
        print(f"Wrote splits to {args.output_dir}")

        print(f"seed={seed}")
        print(f"{'split':<16}{'articles':>9}{'paragraphs':>12}{'questions':>11}{'% questions':>13}")
        total_q = stats(train["data"])["questions"]
        for name in SPLITS:
            s = stats(split_articles[name])
            print(f"{name:<16}{s['articles']:>9}{s['paragraphs']:>12}{s['questions']:>11}"
                f"{100 * s['questions'] / total_q:>12.1f}%")
        d = stats(dev["data"])
        print(f"{'dev (untouched)':<16}{d['articles']:>9}{d['paragraphs']:>12}{d['questions']:>11}")

if __name__ == "__main__":
    main()