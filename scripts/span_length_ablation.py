"""Compare the basic top-20 reranker with and without span length on internal validation."""
import argparse
import json
from zipfile import ZipFile

from offline_handoff import contract
from reranker.core import FeatureConfig, Reranker, questions_from_records
from reranker.real_training import dataset_index, load_scorer, score, targets_from_training
from reranker.team_candidates import prepare_pair


def check_source(batch, raw):
    dataset, index, _ = dataset_index(raw)
    origins = batch['audit']['retained_origins']
    if set(index) != {q['question_id'] for q in origins}:
        raise ValueError('Dataset and candidate IDs differ')
    for q in origins:
        source = index[q['question_id']]
        if (q['article_id'], q['context'], q['question']) != (
                source['article_id'], source['context'], source['question'].strip()):
            raise ValueError(f"Candidate/source mismatch for {q['question_id']}")
    return dataset


def run(data_zip, inference_zip):
    with ZipFile(inference_zip) as archive:
        train, val = prepare_pair(archive.read('inference/reranker_train_candidates.json'),
                                  archive.read('inference/internal_val_candidates.json'), group='basic')
        baseline = contract.decode(archive.read('inference/internal_val_predictions.json'))
    with ZipFile(data_zip) as archive:
        train_data = check_source(train, archive.read('data/splits/reranker_train.json'))
        val_data = check_source(val, archive.read('data/splits/internal_val.json'))
    first = {q['question_id']: q['candidates'][0]['answer'] for q in val['questions']}
    if first != baseline:
        raise ValueError('First candidates differ from baseline predictions')

    evaluate = load_scorer()
    targets = targets_from_training(evaluate, train_data, train['questions'])
    train_questions = questions_from_records(train['questions'])
    val_questions = questions_from_records(val['questions'])
    names = train['feature_registry']['basic']
    base = score(evaluate, val_data, baseline)
    base_em = base['exact_per_question']
    results = {'bert': {k: base[k] for k in ('total', 'exact', 'f1')}}
    fixes, losses = {}, {}
    for label, inputs in [('basic', names), ('without_length', [n for n in names if n != 'span_token_count'])]:
        model = Reranker(FeatureConfig(inputs), alpha=1.0, no_signal='error')
        model.fit(train_questions, targets)
        metrics = score(evaluate, val_data, model.predict(val_questions))
        exact = metrics['exact_per_question']
        fixes[label] = {qid for qid in baseline if exact[qid] > base_em[qid]}
        losses[label] = {qid for qid in baseline if exact[qid] < base_em[qid]}
        results[label] = dict(features=inputs, exact=metrics['exact'], f1=metrics['f1'],
                              corrected=len(fixes[label]), broken=len(losses[label]))
    results['original_errors_avoided'] = len(losses['basic'] - losses['without_length'])
    results['original_fixes_lost'] = len(fixes['basic'] - fixes['without_length'])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Team data.zip')
    parser.add_argument('--inference', required=True, help='Team inference.zip')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.inference), indent=2))
