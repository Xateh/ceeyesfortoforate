"""Gold-aware diagnostic of existing internal-validation candidates; no fitting."""
import argparse
from collections import Counter
from pathlib import Path

from offline_handoff import contract as c
from reranker.real_training import ROOT, SOURCE_FILES, dataset_index, load_scorer, score
from reranker.team_candidates import adapt_export


def diagnose(dataset_bytes, candidate_bytes, baseline_bytes):
    """Return metric upper bounds, retaining supplied order and repository scoring.

    EM and F1 maxima are computed independently per question. Gold labels are
    used for analysis only; oracle selections are not deployable predictions.
    """
    dataset, index, _ = dataset_index(dataset_bytes)
    baseline = c.decode(baseline_bytes)
    batch = adapt_export(candidate_bytes, group='basic', baseline=baseline)
    audit = batch['audit']
    c.require(audit['metadata']['split'] == 'internal_val', 'internal_val candidates required')
    questions = batch['questions']
    c.require(set(index) == set(audit['membership']['questions']), 'candidate/split coverage mismatch')
    for q in audit['retained_origins']:
        source = index[q['question_id']]
        c.require(all(q[k] == source[k] for k in ('article_id', 'context'))
                  and q['question'] == source['question'].strip(), 'candidate/source identity mismatch')

    evaluate = load_scorer()
    baseline_metrics = score(evaluate, dataset, baseline)
    rows = {q['question_id']: {
        'candidate_count': len(q['candidates']),
        'baseline_exact': baseline_metrics['exact_per_question'][q['question_id']],
        'baseline_f1': baseline_metrics['f1_per_question'][q['question_id']],
        'oracle_exact': 0, 'oracle_f1': 0.0, 'first_exact_rank': None,
    } for q in questions}
    curve = []
    # Repeat the last candidate for short groups so every scorer call has exact
    # coverage; update only actual ranks. Every gold/scoring rule stays upstream.
    for rank in range(1, max(row['candidate_count'] for row in rows.values()) + 1):
        predictions = {q['question_id']: q['candidates'][min(rank, len(q['candidates'])) - 1]['answer']
                       for q in questions}
        metrics = baseline_metrics if rank == 1 else score(evaluate, dataset, predictions)
        for qid, row in rows.items():
            if rank > row['candidate_count']:
                continue
            exact, f1 = metrics['exact_per_question'][qid], metrics['f1_per_question'][qid]
            row['oracle_exact'] = max(row['oracle_exact'], exact)
            row['oracle_f1'] = max(row['oracle_f1'], f1)
            if exact and row['first_exact_rank'] is None:
                row['first_exact_rank'] = rank
        curve.append({'k': rank, 'exact': 100 * sum(r['oracle_exact'] for r in rows.values()) / len(rows),
                      'f1': 100 * sum(r['oracle_f1'] for r in rows.values()) / len(rows)})

    errors = sum(not r['baseline_exact'] for r in rows.values())
    recoverable = sum(not r['baseline_exact'] and bool(r['oracle_exact']) for r in rows.values())
    summary = {
        'total': len(rows),
        'baseline': {k: baseline_metrics[k] for k in ('exact', 'f1')},
        'oracle': {k: curve[-1][k] for k in ('exact', 'f1')},
        'headroom_percentage_points': {k: curve[-1][k] - baseline_metrics[k] for k in ('exact', 'f1')},
        'baseline_em_errors': errors,
        'recoverable_em_errors': recoverable,
        'unrecoverable_em_errors': errors - recoverable,
        'recoverable_fraction_of_em_errors': recoverable / errors if errors else None,
        'candidate_count_histogram': dict(sorted(Counter(r['candidate_count'] for r in rows.values()).items())),
    }
    return {
        'schema': 'cs4248.candidate-oracle/1', 'split': 'internal_val',
        'status': 'gold_aware_diagnostic_not_deployable',
        'input_sha256': {'dataset': c.sha256(dataset_bytes), 'candidates': c.sha256(candidate_bytes),
                         'baseline': c.sha256(baseline_bytes)},
        'software_sha256': {name: c.sha256((ROOT / name).read_bytes()) for name in
                           (*SOURCE_FILES, 'scripts/candidate_oracle.py', 'reranker/real_training.py',
                            'reranker/team_candidates.py', 'offline_handoff/contract.py')},
        'generation_metadata': audit['metadata'],
        'baseline_first_candidate_parity': True,
        'source_checkpoint_tokenizer_verified': False,
        'course_scorer_equivalence_verified': False,
        'summary': summary, 'oracle_by_k': curve, 'per_question': rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='SQuAD 1.1 internal_val.json')
    parser.add_argument('--candidates', type=Path, required=True, help='Schema-1 internal_val export')
    parser.add_argument('--baseline', type=Path, required=True, help='Matching baseline predictions')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; parent must exist')
    args = parser.parse_args(argv)
    result = diagnose(args.dataset.read_bytes(), args.candidates.read_bytes(), args.baseline.read_bytes())
    with args.output.open('xb') as handle:
        handle.write(c.encode(result))


if __name__ == '__main__':
    main()
