"""Candidate pooling boundaries and default CLS compatibility, without downloads."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import BertTokenizerFast

from scripts.context_reranker import ContextScorer, batch_scores, encode


class PositionEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, **kwargs):
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        return SimpleNamespace(last_hidden_state=torch.stack((input_ids.float(), positions.float()), dim=-1))


class ContextPoolingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        vocab = Path(cls.directory.name) / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
            '[unused0]', '[unused1]', 'what', '?', 'new', 'york', 'city', 'is', 'large', '.', 'a']), encoding='utf-8')
        cls.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def span(self, context, answer):
        start = context.index(answer)
        return dict(answer=answer, start=start, end=start+len(answer))

    def assert_interval(self, context, answer):
        span = self.span(context, answer)
        inputs, (start, end) = encode(self.tokenizer, 'What?', context, span, return_span=True)
        self.assertEqual(inputs['input_ids'][start:end], self.tokenizer.encode(answer, add_special_tokens=False))
        self.assertEqual(inputs['input_ids'][start-1], self.tokenizer.convert_tokens_to_ids('[unused0]'))
        self.assertEqual(inputs['input_ids'][end], self.tokenizer.convert_tokens_to_ids('[unused1]'))
        self.assertEqual(dict(inputs), dict(encode(self.tokenizer, 'What?', context, span)))
        return inputs, (start, end)

    def test_one_token_answer(self):
        _, (start, end) = self.assert_interval('New York City is large.', 'York')
        self.assertEqual(end-start, 1)

    def test_overlapping_answers(self):
        context = 'New York City is large.'
        self.assert_interval(context, 'New York')
        self.assert_interval(context, 'York City')

    def test_padded_batch_pools_only_candidate_states(self):
        tokenizer = self.tokenizer
        original_side = tokenizer.padding_side
        try:
            for side in ('right', 'left'):
                tokenizer.padding_side = side
                rows = [('short', 'What?', 'York', [self.span('York', 'York')]*2, 0.),
                    ('long', 'What?', 'New York City is large.',
                        [self.span('New York City is large.', a) for a in ('New York', 'York City')], 1.)]
                model = ContextScorer(PositionEncoder(), pooling='span_mean')
                with torch.no_grad():
                    model.head.weight.copy_(torch.tensor([[1., 10.]]))
                captured = []
                handle = model.register_forward_pre_hook(lambda module, args: captured.append(args))
                actual = batch_scores(model, tokenizer, rows, 'cpu').flatten()
                handle.remove()
                batch, spans = captured[0]
                self.assertTrue(bool((batch['attention_mask'] == 0).any()))
                expected = []
                for index, (start, end) in enumerate(spans.tolist()):
                    answer = rows[index//2][3][index%2]['answer']
                    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                    self.assertEqual(batch['input_ids'][index, start:end].tolist(), answer_ids)
                    self.assertTrue(bool(batch['attention_mask'][index, start:end].all()))
                    expected.append(sum(answer_ids)/len(answer_ids) + 10*(start+end-1)/2)
                torch.testing.assert_close(actual, torch.tensor(expected))
        finally:
            tokenizer.padding_side = original_side

    def test_default_cls_and_zero_head(self):
        model = ContextScorer(PositionEncoder())
        self.assertEqual(model.pooling, 'cls')
        batch = dict(input_ids=torch.tensor([[2, 9, 3], [2, 10, 3]]),
                     attention_mask=torch.ones(2, 3, dtype=torch.long))
        self.assertEqual(model(batch).tolist(), [[0., 0.]])
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor([[1., 10.]]))
        self.assertEqual(model(batch).tolist(), [[2., 2.]])

    def test_empty_or_padded_span_is_rejected(self):
        model = ContextScorer(PositionEncoder(), pooling='span_mean')
        batch = dict(input_ids=torch.tensor([[2, 9, 0], [2, 10, 0]]),
                     attention_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]))
        for spans in (torch.tensor([[1, 1], [1, 2]]), torch.tensor([[1, 3], [1, 2]])):
            with self.assertRaisesRegex(ValueError, 'nonempty and exclude padding'):
                model(batch, spans)


if __name__ == '__main__':
    unittest.main()
