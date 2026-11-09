import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast
from scripts.context_reranker import ContextScorer, batch_scores, examples
from reranker.real_training import load_scorer

OUT = Path(__file__).resolve().parent
PILOT = ROOT / 'outputs/context-top2-pilot'
EXPECTED = '64b2b5725953e0a410086239887cf6d03fad787ebe4fcfe48405d16ee79c34ad'


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def hashes():
    result = {}
    for p in PILOT.iterdir():
        if p.is_file():
            with p.open('rb') as stream:
                result[p.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def event(**value):
    value['elapsed_seconds'] = round(time.monotonic() - START, 1)
    text = json.dumps(value)
    print(text, flush=True)
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')


def main():
    assert not (OUT / 'config.json').exists(), 'Fresh output directory required'
    original_hashes = hashes()
    assert original_hashes['weights.pt'] == EXPECTED
    random.seed(4248)
    torch.manual_seed(4248)
    device = torch.device('cuda')
    gpu = torch.cuda.get_device_name(0)
    assert '5060 Ti' in gpu
    evaluate = load_scorer()
    exports = ROOT / 'outputs/issue7-validation-20261104/inference'
    splits = ROOT / 'data/splits'
    train, train_data = examples(exports, splits, 'reranker_train')
    valid, val_data = examples(exports, splits, 'internal_val')
    assert not ({r[0] for r in train} & {r[0] for r in valid})
    base = {r[0]: r[3][0]['answer'] for r in valid}
    assert base == read(exports / 'internal_val_predictions.json')
    labels = [evaluate(train_data, {r[0]: r[3][j]['answer'] for r in train})['f1_per_question'] for j in (0, 1)]
    differences = {i: f1-labels[0][i] for i, f1 in labels[1].items()}
    informative = [r for r in train if differences[r[0]] != 0]
    assert len(informative) == 8751
    # Exactly the original pilot's shuffle, before model construction.
    random.shuffle(informative)
    first_order = [r[0] for r in informative]
    save(OUT / 'epoch-1-order.json', first_order)
    checkpoint = ROOT / 'checkpoints/bert-base-baseline/final'
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    encoder = BertModel.from_pretrained(checkpoint, add_pooling_layer=False, local_files_only=True)
    model = ContextScorer(encoder, pooling='span_mean').to(device)
    assert not torch.count_nonzero(model.head.weight) and not torch.count_nonzero(model.head.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    save(OUT / 'config.json', dict(seed=4248, epochs=5, prespecified_comparison_epoch=5,
        batch_size=4, gpu=gpu, torch_version=torch.__version__, checkpoint=str(checkpoint),
        original_pilot_weights_sha256=EXPECTED, optimizer='AdamW', optimizer_defaults=optimizer.defaults,
        pooling='span_mean', gradient_clip_norm=1., scheduler=None, training_questions=len(train), informative_pairs=8751,
        epoch_order='pilot shuffle at epoch 1; in-place shuffle using continuing Python RNG at epochs 2-5',
        sampling='all 8751 informative pairs once per epoch, no balancing or resampling',
        decision_rule='candidate 2 iff Python-float (score2-score1) > original BERT gap',
        original_pilot_hashes=original_hashes))
    baseline = evaluate(val_data, base)
    results = []

    def infer(rows, phase, epoch):
        scores = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), 4):
                scores.extend(batch_scores(model, tokenizer, rows[start:start+4], device).tolist())
                if start % 4000 == 0:
                    event(epoch=epoch, phase=phase, scored=min(start+4, len(rows)), total=len(rows))
        return scores

    def train_report(scores):
        tensor = torch.tensor(scores, dtype=torch.float32)
        gaps = tensor.new_tensor([r[4] for r in train])
        delta = tensor.new_tensor([differences[r[0]] for r in train])
        margin = tensor[:, 1] - tensor[:, 0] - gaps
        losses = delta.abs() * nn.functional.softplus(-delta.sign() * margin)
        learned = np.array([second-first for first, second in scores])
        exact_delta = np.array([differences[r[0]] for r in train])
        exact_gap = np.array([r[4] for r in train])
        groups = []
        for winner, mask in [(1, exact_delta < 0), (2, exact_delta > 0)]:
            indices = np.flatnonzero(mask)
            groups.append(dict(gold_winner=winner, pairs=int(mask.sum()),
                mean_weighted_pairwise_loss=float(losses[indices].double().mean()),
                median_absolute_learned_difference=float(np.median(np.abs(learned[mask]))),
                switches_to_candidate_2=int((learned[mask] > exact_gap[mask]).sum())))
        assert sum(g['pairs'] for g in groups) == 8751
        return groups

    zero_check = {}
    # Restore RNG state after diagnostics to preserve the training dropout stream.
    with torch.random.fork_rng(devices=[0]):
        for split, rows in [('reranker_train', train), ('internal_val', valid)]:
            initial_scores = infer(rows, f'{split}_zero_head_eval', 0)
            assert all(first == 0 and second == 0 for first,second in initial_scores)
            initial_predictions = {r[0]: r[3][int(second-first > r[4])]['answer']
                for r,(first,second) in zip(rows, initial_scores)}
            bert_predictions = {r[0]: r[3][0]['answer'] for r in rows}
            assert initial_predictions == bert_predictions
            zero_check[split] = dict(questions=len(rows), all_scores_zero=True,
                predictions_identical_to_bert=True)
    save(OUT / 'zero_head_check.json', zero_check)
    event(phase='zero_head_verification_passed', checks=zero_check)

    for epoch in range(1, 6):
        epoch_dir = OUT / f'epoch-{epoch}'
        epoch_dir.mkdir(exist_ok=False)
        if epoch > 1:
            random.shuffle(informative)
            save(epoch_dir / 'training-order.json', [r[0] for r in informative])
        assert len({r[0] for r in informative}) == 8751
        reference_order = (ROOT / 'outputs/context-top2-five-epoch-diagnostic/epoch-1-order.json'
            if epoch == 1 else ROOT / f'outputs/context-top2-five-epoch-diagnostic/epoch-{epoch}/training-order.json')
        assert [r[0] for r in informative] == read(reference_order), 'Epoch order differs from original run'
        assert sum(differences[r[0]] < 0 for r in informative) == 6516
        assert sum(differences[r[0]] > 0 for r in informative) == 2235
        event(epoch=epoch, phase='reference_pair_order_verified', pairs=8751)
        model.train()
        for start in range(0, len(informative), 4):
            batch = informative[start:start+4]
            scores = batch_scores(model, tokenizer, batch, device)
            margin = scores[:, 1] - scores[:, 0] - scores.new_tensor([r[4] for r in batch])
            delta = scores.new_tensor([differences[r[0]] for r in batch])
            loss = (delta.abs() * nn.functional.softplus(-delta.sign() * margin)).mean()
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if start % 1000 == 0:
                event(epoch=epoch, phase='training', visited=min(start+4, len(informative)),
                    total=len(informative), optimizer_step=(epoch-1)*2188+start//4+1)
        torch.save(model.state_dict(), epoch_dir / 'weights.pt')
        val_scores = infer(valid, 'internal_val_eval', epoch)
        predictions = {r[0]: r[3][int(second-first > r[4])]['answer']
            for r,(first,second) in zip(valid,val_scores)}
        measured = evaluate(val_data, predictions)
        be, me = baseline['exact_per_question'], measured['exact_per_question']
        val_report = dict(exact=measured['exact'], f1=measured['f1'],
            changed=sum(predictions[i] != base[i] for i in base),
            fixed=sum(me[i] > be[i] for i in base), broken=sum(me[i] < be[i] for i in base))
        save(epoch_dir / 'internal_val_predictions.json', predictions)
        training_scores = infer(train, 'reranker_train_eval', epoch)
        groups = train_report(training_scores)
        save(epoch_dir / 'reranker_train_scores.json', [dict(question_id=r[0], score_1=s[0], score_2=s[1])
            for r,s in zip(train,training_scores)])
        report = dict(epoch=epoch, optimizer_steps=epoch*2188, train_groups=groups, internal_val=val_report)
        save(epoch_dir / 'metrics.json', report)
        results.append(report)
        save(OUT / 'trajectory.json', results)
        event(epoch=epoch, phase='epoch_complete', report=report)
    torch.save(optimizer.state_dict(), OUT / 'final_optimizer.pt')
    assert hashes() == original_hashes, 'Original pilot artifacts changed'
    previous = read(ROOT / 'outputs/context-top2-five-epoch-diagnostic/summary.json')
    old_final, new_final = previous['trajectory'][-1], results[-1]
    comparison = dict(epoch=5, original_cls=old_final, span_mean=new_final,
        internal_val_delta={k:new_final['internal_val'][k]-old_final['internal_val'][k]
            for k in ('exact','f1','changed','fixed','broken')})
    save(OUT / 'summary.json', dict(pooling='span_mean', prespecified_comparison_epoch=5,
        baseline={k:baseline[k] for k in ('exact','f1')}, trajectory=results,
        zero_head_check=zero_check, epoch_5_comparison=comparison,
        all_epoch_pair_orders_match_reference=True,
        completed_pilot_preserved=True, official_dev_used=False))
    event(phase='complete', output=str(OUT), pilot_preserved=True)


if __name__ == '__main__':
    START = time.monotonic()
    main()
