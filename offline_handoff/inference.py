"""Strict label-free synthetic inference profile; no model or scorer execution."""
from . import contract as c

SCHEMA_ID = 'cs4248.label-free-inference/0.1.0-draft'
FILES = frozenset(('contexts.json', 'candidates.json', 'questions.json',
                   'provenance.json', 'split-manifest.json'))
IDENTITY_FIELDS = ('schema_id', 'contract_id', 'feature_spec_id', 'feature_registry',
                   'candidate_policy', 'core_commit', 'generator_commit',
                   'feature_code_commit', 'checkpoint_sha256', 'tokenizer_id_and_revision')


def unpack(bundle):
    c.require(type(bundle) in (tuple, list) and len(bundle) == 2,
              'bundle requires manifest bytes and artifact dictionary')
    raw, artifacts = bundle
    c.require(type(raw) is bytes and type(artifacts) is dict, 'bundle bytes/dictionary required')
    return raw, artifacts


def identity(manifest):
    """Extract identity from an already validated labelled training manifest."""
    return {name:manifest[name] for name in IDENTITY_FIELDS}


def validate_identity(value):
    c.keys(value, IDENTITY_FIELDS, 'inference identity')
    for name, expected in (('schema_id', c.SCHEMA_ID), ('contract_id', c.CONTRACT_ID),
                           ('feature_spec_id', c.FEATURE_SPEC_ID), ('core_commit', c.CORE_COMMIT)):
        c.require(value[name] == expected, f'incompatible {name}')
    c.validate_config(value['feature_registry'])
    c.validate_policy(value['candidate_policy'])
    for name in ('generator_commit', 'feature_code_commit', 'checkpoint_sha256'):
        c.require(value[name] == c.UNAVAILABLE, f'{name}: synthetic sentinel required')
    c.keys(value['tokenizer_id_and_revision'], ('id', 'revision'), 'tokenizer identity')
    c.require(all(v == c.UNAVAILABLE for v in value['tokenizer_id_and_revision'].values()),
              'synthetic tokenizer sentinel required')


def membership(data):
    """Deterministic exclusion index from validated records; hashes are not attestations."""
    split = data['split-manifest.json']
    values = {'questions': {q['question_id'] for q in data['questions.json']},
              'articles': {a['article_id'] for a in split['articles']},
              'article_hashes': {a['sha256'] for a in split['articles']},
              'contexts': {x['context_id'] for x in data['contexts.json']},
              'context_hashes': {c.sha256(x['text'].encode('utf-8')) for x in data['contexts.json']}}
    return {key:sorted(items) for key,items in values.items()}


def validate_bundle(manifest_bytes, artifacts):
    """Validate only five label-free artifacts and return fresh decoded objects.

    The execution role is separate from the preserved synthetic split status.
    No training/gold artifacts are accepted, constructed or passed to the core.
    """
    c.require(type(manifest_bytes) is bytes, 'manifest bytes required')
    manifest = c.decode(manifest_bytes)
    c.keys(manifest, ('schema_id', 'status', 'execution_role', 'dataset_id', 'identity',
                      'artifact_sha256'), 'label-free manifest')
    c.require(manifest['schema_id'] == SCHEMA_ID, 'label-free schema mismatch')
    c.require(manifest['status'] == c.SYNTHETIC, 'only synthetic inference is enabled')
    c.require(manifest['execution_role'] == 'internal_validation', 'forbidden inference role')
    c.identifier(manifest['dataset_id'], 'inference dataset_id')
    validate_identity(manifest['identity'])
    c.keys(artifacts, FILES, 'strict label-free artifact set')
    c.keys(manifest['artifact_sha256'], FILES, 'label-free artifact hash map')
    for name, raw in artifacts.items():
        c.require(type(raw) is bytes, 'artifact bytes required')
        c.require(manifest['artifact_sha256'][name] == c.sha256(raw), f'artifact hash mismatch: {name}')
    data = {name:c.decode(raw) for name,raw in artifacts.items()}
    config = manifest['identity']['feature_registry']
    expected, provenance = c.adapt_candidates(data['contexts.json'], data['candidates.json'],
                                              config, manifest['identity']['candidate_policy'])
    c.validate_questions(data['questions.json'], expected, config)
    c.require(c.encode(data['provenance.json']) == c.encode(provenance),
              'provenance differs from complete source origins/winners')
    c.validate_splits(data['split-manifest.json'], data['contexts.json'], data['candidates.json'],
                      'synthetic', True)
    return manifest, data
