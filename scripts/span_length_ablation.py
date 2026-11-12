"""Compare the basic top-20 reranker with and without span length on internal validation."""
import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

from offline_handoff import contract
from reranker.core import FeatureConfig, Reranker, questions_from_records
from reranker.real_training import dataset_index, load_scorer, score, targets_from_training
from reranker.team_candidates import prepare_pair


LIST_QUESTION = re.compile(r'\b(two|three|four|five|both)\b', re.I)
CONJUNCTION = re.compile(r'\b(and|or)\b', re.I)
LIST_FEATURE = 'list_question_and_coordinated_answer'


def add_list_feature(batch):
    """Candidate-local proxy for list completeness, computed without gold answers."""
    origins = batch['audit']['retained_origins']
    for record, origin in zip(batch['questions'], origins, strict=True):
        assert record['question_id'] == origin['question_id']
        asks_for_list = bool(LIST_QUESTION.search(origin['question']))
        for candidate in record['candidates']:
            candidate['features'][LIST_FEATURE] = int(asks_for_list and bool(CONJUNCTION.search(candidate['answer'])))


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


def run(data_zip, inference_zip, output=None):
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
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
    add_list_feature(train)
    add_list_feature(val)
    train_questions = questions_from_records(train['questions'])
    val_questions = questions_from_records(val['questions'])
    names = train['feature_registry']['basic']
    base = score(evaluate, val_data, baseline)
    base_em = base['exact_per_question']
    results = {'bert': {k: base[k] for k in ('total', 'exact', 'f1')}}
    fixes, losses = {}, {}
    reduced = [n for n in names if n != 'span_token_count']
    discovery_losses = set()
    predictions = {}
    for label, inputs in [('basic', names), ('without_length', reduced), ('with_list', reduced + [LIST_FEATURE])]:
        model = Reranker(FeatureConfig(inputs), alpha=1.0, no_signal='error')
        model.fit(train_questions, targets)
        selected = model.select(val_questions)
        predictions[label] = {qid: candidate.answer for qid, candidate in selected.items()}
        if output is not None:
            model.save(output / (label + '-model.json'))
            (output / (label + '-predictions.json')).write_text(json.dumps(predictions[label], ensure_ascii=False), encoding='utf-8')
        metrics = score(evaluate, val_data, predictions[label])
        exact = metrics['exact_per_question']
        fixes[label] = {qid for qid in baseline if exact[qid] > base_em[qid]}
        losses[label] = {qid for qid in baseline if exact[qid] < base_em[qid]}
        results[label] = dict(features=inputs, exact=metrics['exact'], f1=metrics['f1'],
                              corrected=len(fixes[label]), broken=len(losses[label]))
        if label == 'basic':
            for record, origin in zip(val['questions'], val['audit']['retained_origins'], strict=True):
                qid = record['question_id']
                if qid not in losses[label] or not LIST_QUESTION.search(origin['question']):
                    continue
                first = origin['candidates'][0]
                for candidate, span in zip(record['candidates'], origin['candidates'], strict=True):
                    if candidate['candidate_id'] == selected[qid].candidate_id:
                        if (first['start'] <= span['start'] and span['end'] <= first['end']
                                and CONJUNCTION.search(first['answer']) and not CONJUNCTION.search(span['answer'])):
                            discovery_losses.add(qid)
                        break
    control, variant = 'without_length', 'with_list'
    results['list_comparison'] = {
        'changed_answers': sum(predictions[control][i] != predictions[variant][i] for i in baseline),
        'em_gains_over_control': len(fixes[variant] - fixes[control]) + len(losses[control] - losses[variant]),
        'em_losses_vs_control': len(fixes[control] - fixes[variant]) + len(losses[variant] - losses[control]),
        'original_list_trim_losses': len(discovery_losses),
        'exact_on_original_list_trim_losses': {
            name: len(discovery_losses - losses[name]) for name in (control, variant)},
    }
    results['original_errors_avoided'] = len(losses['basic'] - losses['without_length'])
    results['original_fixes_lost'] = len(fixes['basic'] - fixes['without_length'])
    if output is not None:
        (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Team data.zip')
    parser.add_argument('--inference', required=True, help='Team inference.zip')
    parser.add_argument('--output', type=Path, help='Fresh directory for models, predictions and metrics')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.inference, args.output), indent=2))
