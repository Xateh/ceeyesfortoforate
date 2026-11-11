"""One audited source transition; keep the original frozen binding bytes intact.

This compatibility rule applies only to the loader source, never model artifacts,
other source files, or arbitrary future edits. New training binds current sources.
"""

LEGACY_LOADER_SHA256 = 'e1c7fc0dfde427cd2ddde8ddd9b8a060b54d9f4405a2b017ed3313ba49c4fe8e'
CORRECTED_LOADER_SHA256 = '121b7b1151e57b7f9affac8d4f133f778a4deaa706d564db64459b08bb7e710e'


def compatible_reload_source(name, expected, actual):
    return (name == 'reranker/real_training.py'
            and expected == LEGACY_LOADER_SHA256
            and actual == CORRECTED_LOADER_SHA256)
