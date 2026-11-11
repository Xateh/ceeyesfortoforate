"""Top-two BERT reranking pilot on the train-derived splits."""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from torch import nn
from transformers import BertModel, BertTokenizerFast

from reranker.real_training import dataset_index, load_scorer


def examples(directory, gold_dir, split):
    payload = json.loads((directory / f'{split}_candidates.json').read_bytes())
    if payload['metadata']['split'] != split or payload['schema_version'] != 1:
        raise ValueError(f'Wrong candidate export for {split}')
    data, index, _ = dataset_index((gold_dir / f'{split}.json').read_bytes())
    rows = []
    seen = set()
    for q in payload['questions']:
        i = q['question_id']
        item = index[i]
        if i in seen or q['context'] != item['context'] or q['question'] != item['question'].strip() or q['article_id'] != item['article_id']:
            raise ValueError(f'Source identity mismatch: {i}')
        seen.add(i)
        spans = q['candidates'][:2]
        if len(spans) != 2:
            raise ValueError(f'Expected two candidates in the team export: {i}')
        for span in spans:
            if q['context'][span['start']:span['end']] != span['answer']:
                raise ValueError(f'Candidate offset mismatch: {i}')
        gap = spans[0]['combined_score'] - spans[1]['combined_score']
        if gap < 0:
            raise ValueError(f'Unordered candidates: {i}')
        rows.append((i, q['question'], q['context'], spans, gap))
    if seen != set(index):
        raise ValueError('Incomplete candidate export')
    return rows, data['data']


def encode(tokenizer, question, context, span, return_span=False):
    # Reserve space for the entire candidate and its two boundary markers.
    ids = lambda s: tokenizer.encode(s, add_special_tokens=False)
    q = ids(question)[:64]
    answer = ids(span['answer'])
    budget = 192 - len(q) - len(answer) - 5  # CLS, SEP x2, unused markers x2
    if budget < 0:
        raise ValueError('Candidate exceeds input budget')
    left = re.split(r'(?<=[.!?])\s+', context[:span['start']])[-1]
    right = re.split(r'(?<=[.!?])\s+', context[span['end']:])[0]
    left, right = ids(left[-1600:]), ids(right[:1600])
    nl = min(len(left), max(budget // 2, budget - len(right)))
    window = (left[-nl:] if nl else []) + [tokenizer.convert_tokens_to_ids('[unused0]')] + answer
    window += [tokenizer.convert_tokens_to_ids('[unused1]')] + right[:budget-nl]
    inputs = tokenizer.prepare_for_model(q, window, return_attention_mask=True, verbose=False)
    if return_span:
        # [CLS] question [SEP] left [unused0] answer [unused1] right [SEP].
        # Record the answer's half-open interval, rather than searching token IDs.
        start = len(q) + 3 + nl
        return inputs, (start, start + len(answer))
    return inputs


class ContextScorer(nn.Module):
    def __init__(self, encoder, pooling='cls'):
        super().__init__()
        if pooling not in ('cls', 'span_mean'):
            raise ValueError(f'Unknown pooling mode: {pooling}')
        self.encoder = encoder
        self.pooling = pooling
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, batch, span_positions=None):
        hidden = self.encoder(**batch).last_hidden_state
        if self.pooling == 'cls':
            pooled = hidden[:, 0]
        else:
            if span_positions is None:
                raise ValueError('span_mean requires recorded candidate positions')
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            mask = (positions >= span_positions[:, :1]) & (positions < span_positions[:, 1:])
            mask &= batch['attention_mask'].bool()
            lengths = mask.sum(dim=1)
            if torch.any(lengths == 0) or torch.any(lengths != span_positions[:, 1] - span_positions[:, 0]):
                raise ValueError('Candidate token interval must be nonempty and exclude padding')
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        return self.head(pooled).reshape(-1, 2)


def batch_scores(model, tokenizer, rows, device):
    if model.pooling == 'span_mean':
        encoded = [encode(tokenizer, question, context, span, return_span=True)
                   for _, question, context, spans, _ in rows for span in spans]
        inputs, spans = zip(*encoded)
        batch = tokenizer.pad(list(inputs), padding=True, return_tensors='pt').to(device)
        offsets = [batch['input_ids'].shape[1] - len(item['input_ids'])
                   if tokenizer.padding_side == 'left' else 0 for item in inputs]
        span_positions = torch.tensor([(start+offset, end+offset)
                                       for (start, end), offset in zip(spans, offsets)], device=device)
        return model(batch, span_positions)
    inputs = [encode(tokenizer, question, context, span)
              for _, question, context, spans, _ in rows for span in spans]
    batch = tokenizer.pad(inputs, padding=True, return_tensors='pt').to(device)
    return model(batch)


def run(args):
    random.seed(4248)
    torch.manual_seed(4248)
    device = torch.device(args.device)
    evaluate = load_scorer()
    train, train_data = examples(args.exports, args.splits, 'reranker_train')
    valid, val_data = examples(args.exports, args.splits, 'internal_val')
    if {r[0] for r in train} & {r[0] for r in valid}:
        raise ValueError('Training/validation overlap')
    base = {r[0]: r[3][0]['answer'] for r in valid}
    supplied = json.loads((args.exports / 'internal_val_predictions.json').read_bytes())
    if base != supplied:
        raise ValueError('Rank one differs from supplied validation baseline')
    labels = [evaluate(train_data, {r[0]: r[3][j]['answer'] for r in train})['f1_per_question']
              for j in (0, 1)]
    differences = {i: f1-labels[0][i] for i, f1 in labels[1].items()}
    informative = [r for r in train if differences[r[0]] != 0]
    if not informative:
        raise ValueError('No unequal training pairs')
    random.shuffle(informative)
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer or args.checkpoint)
    encoder = BertModel.from_pretrained(args.checkpoint, add_pooling_layer=False)
    model = ContextScorer(encoder, pooling=args.pooling).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for start in range(0, len(informative), args.batch_size):
        rows = informative[start:start+args.batch_size]
        scores = batch_scores(model, tokenizer, rows, device)
        margin = scores[:, 1] - scores[:, 0] - scores.new_tensor([r[4] for r in rows])
        delta = scores.new_tensor([differences[r[0]] for r in rows])
        loss = (delta.abs() * nn.functional.softplus(-delta.sign() * margin)).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    torch.save(model.state_dict(), args.output / 'weights.pt')
    model.eval()
    preds = {}
    with torch.inference_mode():
        for start in range(0, len(valid), args.batch_size):
            rows = valid[start:start+args.batch_size]
            for row, (first, second) in zip(rows, batch_scores(model, tokenizer, rows, device).tolist()):
                preds[row[0]] = row[3][int(second-first > row[4])]['answer']
    b, m = evaluate(val_data, base), evaluate(val_data, preds)
    be, me = b['exact_per_question'], m['exact_per_question']
    result = dict(seed=4248, batch_size=args.batch_size, pooling=args.pooling, training_questions=len(train),
                  informative_pairs=len(informative), validation_questions=len(valid),
                  baseline={k:b[k] for k in ('exact','f1')}, context={k:m[k] for k in ('exact','f1')},
                  changed=sum(preds[i]!=base[i] for i in base),
                  fixed=sum(me[i]>be[i] for i in base), broken=sum(me[i]<be[i] for i in base))
    (args.output/'predictions.json').write_text(json.dumps(preds), encoding='utf-8')
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', type=Path, required=True)
    p.add_argument('--splits', type=Path, default=Path('data/splits'))
    p.add_argument('--checkpoint', default='checkpoints/bert-base-baseline/final')
    p.add_argument('--tokenizer', help='Matching tokenizer directory if stored separately')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4, help='Questions per batch (two sequences each)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--pooling', choices=('cls', 'span_mean'), default='cls',
                   help='Candidate representation for the correction head (default: cls)')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('Positive batch size required')
    run(args)
