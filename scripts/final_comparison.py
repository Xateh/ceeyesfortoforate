"""Replay the frozen official-dev experiment with the two original ZIP inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from offline_handoff.contract import features
from reranker.core import Reranker, questions_from_records
from scripts.squad_eval import evaluate

GAP = 0.02748394012451172
MODEL = Path(__file__).resolve().parents[1] / 'configs/frozen-basic-model.json'


def read_input(path, member, digest):
    with ZipFile(path) as archive:
        raw = archive.read(member)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f'{member} differs from the recorded final-run input')
    return json.loads(raw)


def predict(export, model):
    questions = export['questions']
    records = []
    for question in questions:
        spans = question['candidates'][:2]
        records.append({
            'question_id': question['question_id'],
            'candidates': [
                {'candidate_id': str(rank), 'answer': span['answer'],
                 'features': features(question['question'], span, 'basic')}
                for rank, span in enumerate(spans)
            ],
        })
    baseline, frozen = {}, {}
    for question, scores in zip(questions, model.scores(questions_from_records(records))):
        first, second = question['candidates'][:2]
        gap = first['combined_score'] - second['combined_score']
        qid = question['question_id']
        baseline[qid] = first['answer']
        frozen[qid] = second['answer'] if gap <= GAP and scores[1] > scores[0] else first['answer']
    return baseline, frozen


def compare(articles, baseline, frozen):
    bert = evaluate(articles, baseline)
    rule = evaluate(articles, frozen)
    changes = [rule['exact_per_question'][i] - bert['exact_per_question'][i] for i in baseline]
    gains, losses = changes.count(1), changes.count(-1)
    discordant = gains + losses
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    article_sizes, article_deltas = [], []
    for article in articles:
        ids = [q['id'] for p in article['paragraphs'] for q in p['qas']]
        article_sizes.append(len(ids))
        article_deltas.append(sum(rule['f1_per_question'][i] - bert['f1_per_question'][i] for i in ids))
    draws = np.random.default_rng(4248).integers(len(articles), size=(20000, len(articles)))
    bootstrap = 100 * np.array(article_deltas)[draws].sum(1) / np.array(article_sizes)[draws].sum(1)
    return {
        'total': bert['total'],
        'baseline': {key: bert[key] for key in ('exact', 'f1')},
        'frozen': {key: rule[key] for key in ('exact', 'f1')},
        'changed': sum(baseline[i] != frozen[i] for i in baseline),
        'exact_gains': gains, 'exact_losses': losses,
        'exact_mcnemar_two_sided_p': min(1, 2 * tail / 2**discordant),
        'f1_delta_pp_article_bootstrap_95ci': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--predictions-dir', type=Path, help='Optional new directory for both prediction maps')
    args = parser.parse_args()
    export = read_input(args.candidates, 'dev_candidates.json',
                        '46524095a11587a8b10195e6f9131eec81b92d343a683386d81a216d38ac1de9')
    baseline, frozen = predict(export, Reranker.load(MODEL))
    dataset = read_input(args.data, 'data/dev-v1.1.json',
                         '95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9')
    result = compare(dataset['data'], baseline, frozen)
    if args.predictions_dir:
        args.predictions_dir.mkdir(parents=True, exist_ok=False)
        for name, predictions in [('baseline', baseline), ('frozen', frozen)]:
            (args.predictions_dir / f'{name}-predictions.json').write_text(
                json.dumps(predictions, ensure_ascii=False), encoding='utf-8')
        (args.predictions_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
