"""Synthetic PR28 model persistence with checked adapter and training bindings."""
from pathlib import Path
import re
import tempfile

from offline_handoff import contract as c
from . import team_candidates as team
from .core import FeatureConfig, Reranker, questions_from_records

SCHEMA = 'cs4248.pr28-reranker-model/0.1.0-draft'
STATUS = 'synthetic_fit_not_production_valid'
TARGET_STATUS = 'synthetic_hand_authored_not_scorer_outputs'
GENERATION = ('checkpoint', 'max_answer_length', 'n_best', 'max_candidates', 'logit_dtype',
              'tie_break', 'max_context_only', 'numpy_version', 'torch_version', 'device')


def _digest(value):
    c.require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'SHA-256 required')


def _identity(batch):
    return {'adapter_profile': team.PROFILE, 'candidate_schema_version': 1,
            'inspected_producer_ref': team.PRODUCER_REF, 'core_commit': c.CORE_COMMIT,
            'feature_registry': batch['feature_registry'],
            'declared_generation': {k: batch['audit']['metadata'][k] for k in GENERATION},
            'provenance_status': 'declared_paths_only_no_checkpoint_or_tokenizer_hash'}


class TeamCandidateModel:
    """New batches are accepted; reload requires original label-free training bytes.

    Python callers and the independently retained model-file pin are trusted.
    Target values are never serialized or read by load/predict. Metadata identity
    is checked, not authenticated. This profile does not enable real-label fits.
    """
    def __init__(self, training_raw, *, group):
        training = team.adapt_export(training_raw, group=group)
        c.require(training['audit']['metadata']['split'] == 'reranker_train', 'forbidden training role')
        config = training['feature_registry']; c.validate_config(config)
        self._features = FeatureConfig(config['basic'], config['compatibility'], config['group'])
        self._records = training['questions']
        self._binding = {'identity': _identity(training),
                         'training_export_sha256': training['audit']['input_sha256'],
                         'training_metadata': training['audit']['metadata'],
                         'training_membership': training['audit']['membership']}
        self._model = Reranker(self._features, no_signal='error')
        self._targets_sha256 = None

    @classmethod
    def fit_synthetic(cls, training_raw, targets, *, group, target_status, alpha=1.0):
        c.require(target_status == TARGET_STATUS, 'synthetic target attestation required')
        run = cls(training_raw, group=group)
        model = Reranker(run._features, alpha=alpha, no_signal='error')
        model.fit(questions_from_records(run._records), targets)
        # Identity only: no labels enter the saved model envelope or inference.
        run._targets_sha256 = c.sha256(c.encode(targets))
        run._model = model
        return run

    def _batch(self, raw):
        batch = team.adapt_export(raw, group=self._features.group)
        c.require(batch['audit']['metadata']['split'] == 'internal_val', 'forbidden inference role')
        c.require(_identity(batch) == self._binding['identity'], 'adapter/feature/generation identity mismatch')
        for key, values in batch['audit']['membership'].items():
            c.require(not set(values) & set(self._binding['training_membership'][key]),
                      f'training/inference {key} overlap')
        return batch

    def scores(self, raw):
        return self._model.scores(questions_from_records(self._batch(raw)['questions']))

    def select(self, raw):
        return self._model.select(questions_from_records(self._batch(raw)['questions']))

    def predict(self, raw):
        batch = self._batch(raw)
        predictions = self._model.predict(questions_from_records(batch['questions']))
        c.validate_mapping(predictions, batch['questions'])
        return predictions

    def predict_with_receipt(self, raw):
        predictions = self.predict(raw)
        return predictions, {'schema_id': SCHEMA, 'status': STATUS,
                             'training_export_sha256': self._binding['training_export_sha256'],
                             'inference_export_sha256': c.sha256(raw),
                             'predictions_sha256': c.sha256(c.encode(predictions)),
                             'model_state_sha256': c.sha256(c.encode(self._core_artifact())),
                             'adapter_identity_sha256': c.sha256(_canonical(self._binding['identity']).encode('utf-8')),
                             'question_count': len(predictions), 'origin_authenticated': False}

    def _core_artifact(self):
        with tempfile.TemporaryDirectory(prefix='team-core-save-') as tmp:
            path = Path(tmp)/'core.json'
            self._model.save(path)
            return c.decode(path.read_bytes())

    def save(self, path):
        """Create a new artifact only; retain the returned digest independently."""
        model = self._core_artifact(); _digest(self._targets_sha256)
        raw = c.encode({'schema_id': SCHEMA, 'status': STATUS, 'binding': self._binding,
                        'target_status': TARGET_STATUS, 'training_targets_sha256': self._targets_sha256,
                        'model': model, 'model_sha256': c.sha256(c.encode(model))})
        with Path(path).open('xb') as out:
            out.write(raw)
        return c.sha256(raw)

    @classmethod
    def load(cls, path, training_raw, *, expected_sha256):
        """Restore state without fitting; expected_sha256 must be retained separately."""
        _digest(expected_sha256)
        raw = Path(path).read_bytes()
        c.require(c.sha256(raw) == expected_sha256, 'model artifact pin mismatch')
        value = c.decode(raw)
        c.keys(value, ('schema_id', 'status', 'binding', 'target_status',
                       'training_targets_sha256', 'model', 'model_sha256'), 'team model envelope')
        c.require(value['schema_id'] == SCHEMA and value['status'] == STATUS and
                  value['target_status'] == TARGET_STATUS, 'model schema/status mismatch')
        binding = value['binding']
        c.keys(binding, ('identity', 'training_export_sha256', 'training_metadata',
                         'training_membership'), 'training binding')
        c.require(type(binding['identity']) is dict, 'model identity required')
        config = binding['identity'].get('feature_registry'); c.validate_config(config)
        run = cls(training_raw, group=config['group'])
        # Canonical JSON comparison distinguishes booleans from integer versions,
        # while allowing harmless JSON-object key order differences.
        c.require(_canonical(binding) == _canonical(run._binding), 'training/adapter identity binding mismatch')
        _digest(value['training_targets_sha256']); _digest(value['model_sha256'])
        model = value['model']
        c.require(value['model_sha256'] == c.sha256(c.encode(model)), 'core state hash mismatch')
        c.require(type(model) is dict and model.get('features') == run._features.to_dict(),
                  'core feature configuration mismatch')
        c.require(model.get('no_signal') == 'error', 'diagnostic fallback forbidden')
        with tempfile.TemporaryDirectory(prefix='team-core-load-') as tmp:
            core_path = Path(tmp)/'core.json'; core_path.write_bytes(c.encode(model))
            restored = Reranker.load(core_path)
        c.require(model['state']['training_questions'] == len(run._records), 'training count mismatch')
        run._targets_sha256 = value['training_targets_sha256']
        run._model = restored
        return run


def _canonical(value):
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
