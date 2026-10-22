"""Synthetic trained-model boundary for independently hashed, label-free batches."""
import copy
from pathlib import Path
import tempfile

from offline_handoff import contract as c
from offline_handoff import inference as boundary
from .core import FeatureConfig, Reranker, questions_from_records

PROFILE = 'cs4248.synthetic-heldout-reranker/0.1.0-draft'
STATUS = 'synthetic_fit_not_production_valid'


class HeldoutRun:
    """Bind training once; accept new internal-validation batches without labels.

    Reload requires the original labelled training bundle to verify provenance,
    but does not fit. SyntheticRun and its exact two-bundle replay are unchanged.
    Python callers and upstream provenance assertions remain trusted.
    """
    def __init__(self, training_bundle):
        raw, artifacts = boundary.unpack(training_bundle)
        c.validate_bundle(raw, artifacts)
        manifest = c.decode(raw)
        c.require(manifest['status'] == c.SYNTHETIC, 'only synthetic training is enabled')
        data = {name:c.decode(value) for name,value in artifacts.items()}
        identity = boundary.identity(manifest)
        boundary.validate_identity(identity)
        self._binding = {'identity':identity, 'training_manifest':manifest,
                         'training_manifest_sha256':c.sha256(raw),
                         'training_membership':boundary.membership(data), 'training_role':'reranker_train'}
        self._records = data['questions.json']
        self._targets = data['targets.json']
        config = identity['feature_registry']
        self._features = FeatureConfig(tuple(config['basic']), tuple(config['compatibility']), config['group'])
        c.require(self._features.names == c.validate_config(config), 'core feature order mismatch')
        self._model = Reranker(self._features, no_signal='error')

    def fit(self, *, role='reranker_train', alpha=1.0):
        c.validate_role(role, 'fit_boundary')
        fitted = Reranker(self._features, alpha=alpha, no_signal='error')
        fitted.fit(questions_from_records(copy.deepcopy(self._records)), copy.deepcopy(self._targets))
        self._model = fitted  # A failed fit preserves the previous model.
        return self

    def _batch(self, bundle):
        raw, artifacts = boundary.unpack(bundle)
        manifest, data = boundary.validate_bundle(raw, artifacts)
        c.require(manifest['identity'] == self._binding['identity'], 'training/inference identity mismatch')
        for key, values in boundary.membership(data).items():
            c.require(not set(values) & set(self._binding['training_membership'][key]),
                      f'training/inference {key} overlap')
        return raw, manifest, data['questions.json']

    def scores(self, bundle):
        _, _, records = self._batch(bundle)
        return self._model.scores(questions_from_records(records))

    def select(self, bundle):
        _, _, records = self._batch(bundle)
        return self._model.select(questions_from_records(records))

    def _predict(self, bundle):
        raw, manifest, records = self._batch(bundle)
        predictions = self._model.predict(questions_from_records(records))
        c.validate_mapping(predictions, records)
        return raw, manifest, predictions

    def predict(self, bundle):
        return self._predict(bundle)[2]

    def _core_artifact(self):
        with tempfile.TemporaryDirectory(prefix='heldout-core-') as tmp:
            path = Path(tmp)/'core.json'
            self._model.save(path)
            return c.decode(path.read_bytes())

    def save_predictions(self, path, bundle):
        """Write actual answer mapping; return a separately retainable batch receipt."""
        raw, manifest, predictions = self._predict(bundle)
        encoded = c.encode(predictions)
        model = self._core_artifact()
        receipt = {'profile_id':PROFILE, 'status':STATUS, 'production_valid':False,
                   'execution_role':manifest['execution_role'],
                   'training_manifest_sha256':self._binding['training_manifest_sha256'],
                   'model_sha256':c.sha256(c.encode(model)),
                   'inference_manifest_sha256':c.sha256(raw),
                   'artifact_sha256':manifest['artifact_sha256'],
                   'predictions_sha256':c.sha256(encoded), 'question_count':len(predictions)}
        Path(path).write_bytes(encoded)
        return receipt

    def save(self, path):
        model = self._core_artifact()
        value = {'schema_id':PROFILE, 'status':STATUS, 'binding':self._binding,
                 'model':model, 'model_sha256':c.sha256(c.encode(model))}
        Path(path).write_bytes(c.encode(value))

    @classmethod
    def load(cls, path, training_bundle):
        """Verify training binding and restore core state, without fitting or validation data."""
        run = cls(training_bundle)
        value = c.decode(Path(path).read_bytes())
        c.keys(value, ('schema_id', 'status', 'binding', 'model', 'model_sha256'), 'heldout model')
        c.require(value['schema_id'] == PROFILE and value['status'] == STATUS, 'heldout model schema/status mismatch')
        c.require(value['binding'] == run._binding, 'training provenance/identity binding mismatch')
        model = value['model']
        c.require(value['model_sha256'] == c.sha256(c.encode(model)), 'core artifact hash mismatch')
        c.require(type(model) is dict and model.get('features') == run._features.to_dict(),
                  'saved core feature identity mismatch')
        c.require(model.get('no_signal') == 'error', 'diagnostic fallback cannot enter normal comparisons')
        with tempfile.TemporaryDirectory(prefix='heldout-core-load-') as tmp:
            core_path = Path(tmp)/'core.json'
            core_path.write_bytes(c.encode(model))
            restored = Reranker.load(core_path)
        informative = sum(len(set(run._targets[q['question_id']].values())) > 1 for q in run._records)
        c.require(model['state']['training_questions'] == len(run._records) and
                  model['state']['informative_questions'] == informative,
                  'saved core training counts differ from bound targets')
        run._model = restored
        return run
