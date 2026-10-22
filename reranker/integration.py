"""Synthetic-only execution profile over unchanged PR20 bundles and PR19 core.

The role assignment is an explicit local experiment input, not a production
split or a change to the handoff's synthetic provenance status.
"""
import copy
from pathlib import Path
import tempfile

from offline_handoff import contract as c
from .core import FeatureConfig, Reranker, questions_from_records

PROFILE = 'cs4248.synthetic-reranker-run/0.1.0-draft'
ROLES = ('reranker_train', 'internal_validation')
IDENTITY_FIELDS = ('schema_id', 'contract_id', 'feature_spec_id', 'feature_registry',
                   'candidate_policy', 'core_commit', 'generator_commit',
                   'feature_code_commit', 'checkpoint_sha256', 'tokenizer_id_and_revision')


def _identity(manifest):
    return {name: manifest[name] for name in IDENTITY_FIELDS}


class SyntheticRun:
    """Run only validated synthetic bundles with explicit, disjoint role assignments.

    No diagnostic no-signal fallback is exposed. Python callers remain trusted;
    this is a misuse guard and integrity envelope, not a security sandbox.
    """
    def __init__(self, bundles):
        c.keys(bundles, ROLES, 'synthetic execution assignments')
        self._records, self._targets, bindings = {}, {}, {}
        used = {key:set() for key in ('questions', 'articles', 'content', 'contexts')}
        identity = None
        for role in ROLES:
            pair = bundles[role]
            c.require(type(pair) in (tuple, list) and len(pair) == 2, 'bundle requires manifest bytes and artifacts')
            raw, artifacts = pair
            c.validate_bundle(raw, artifacts)
            manifest = c.decode(raw)
            c.require(manifest['status'] == c.SYNTHETIC, 'only synthetic execution is enabled')
            current_identity = _identity(manifest)
            if identity is None:
                identity = current_identity
            c.require(current_identity == identity, 'training/validation feature or provenance identity mismatch')
            data = {name:c.decode(value) for name,value in artifacts.items()}
            records = data['questions.json']
            split = data['split-manifest.json']
            sets = {'questions':{q['question_id'] for q in records},
                    'articles':{a['article_id'] for a in split['articles']},
                    'content':{a['sha256'] for a in split['articles']},
                    'contexts':{context['text'] for context in data['contexts.json']}}
            for key, values in sets.items():
                c.require(not used[key] & values, f'cross-role {key} overlap')
                used[key].update(values)
            self._records[role] = records
            self._targets[role] = data['targets.json']
            bindings[role] = {'manifest_sha256':c.sha256(raw), 'manifest':manifest}
        self._binding = {'profile_id':PROFILE, 'identity':identity, 'bundles':bindings}
        config = identity['feature_registry']
        self._features = FeatureConfig(tuple(config['basic']), tuple(config['compatibility']), config['group'])
        c.require(self._features.names == c.validate_config(config), 'core feature order mismatch')
        self._model = Reranker(self._features, no_signal='error')

    def _questions(self, role):
        c.require(type(role) is str and role in ROLES, 'unsupported synthetic execution role')
        return questions_from_records(copy.deepcopy(self._records[role]))

    def fit(self, *, role='reranker_train', alpha=1.0):
        c.validate_role(role, 'fit_boundary')
        questions = self._questions(role)
        # Failed refits leave the previous successful model intact.
        fitted = Reranker(self._features, alpha=alpha, no_signal='error')
        fitted.fit(questions, copy.deepcopy(self._targets[role]))
        self._model = fitted
        return self

    def scores(self, *, role='internal_validation'):
        return self._model.scores(self._questions(role))

    def select(self, *, role='internal_validation'):
        return self._model.select(self._questions(role))

    def predict(self, *, role='internal_validation'):
        questions = self._questions(role)
        predictions = self._model.predict(questions)
        c.keys(predictions, [q.question_id for q in questions], 'prediction coverage')
        c.validate_mapping(predictions, self._records[role])
        return predictions

    def save_predictions(self, path, *, role='internal_validation'):
        # Actual model predictions, never the bundle's illustration mapping.
        Path(path).write_bytes(c.encode(self.predict(role=role)))

    def save(self, path):
        with tempfile.TemporaryDirectory(prefix='synthetic-core-save-') as tmp:
            core_path = Path(tmp) / 'core.json'
            self._model.save(core_path)
            model = c.decode(core_path.read_bytes())
        value = {'schema_id':PROFILE, 'status':'synthetic_fit_not_production_valid',
                 'binding':self._binding, 'model':model, 'model_sha256':c.sha256(c.encode(model))}
        Path(path).write_bytes(c.encode(value))

    @classmethod
    def load(cls, path, bundles):
        """Replay against the same bound bundle bytes; not arbitrary-data inference."""
        run = cls(bundles)
        value = c.decode(Path(path).read_bytes())
        c.keys(value, ('schema_id', 'status', 'binding', 'model', 'model_sha256'), 'saved synthetic run')
        c.require(value['schema_id'] == PROFILE and value['status'] == 'synthetic_fit_not_production_valid',
                  'unsupported synthetic run artifact')
        c.require(value['binding'] == run._binding, 'contract/feature/order/provenance binding mismatch')
        model = value['model']
        c.require(value['model_sha256'] == c.sha256(c.encode(model)), 'core artifact hash mismatch')
        c.require(type(model) is dict and model.get('features') == run._features.to_dict(),
                  'saved core feature identity mismatch')
        c.require(model.get('no_signal') == 'error', 'diagnostic fallback cannot enter normal comparisons')
        with tempfile.TemporaryDirectory(prefix='synthetic-core-load-') as tmp:
            core_path = Path(tmp) / 'core.json'
            core_path.write_bytes(c.encode(model))
            restored = Reranker.load(core_path)
        # Bind core training counts too; hashes are integrity checks, not attestations.
        records = run._records['reranker_train']
        targets = run._targets['reranker_train']
        informative = sum(len(set(targets[q['question_id']].values())) > 1 for q in records)
        c.require(model['state']['training_questions'] == len(records) and
                  model['state']['informative_questions'] == informative,
                  'saved core training counts differ from bound targets')
        run._model = restored
        return run
