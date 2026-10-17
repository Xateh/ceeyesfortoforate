# Reranker synthetic validation — 17 October 2026

## Scope and provenance

Work ran on xlogin1 in
`/home/t/tehxa/cs4248-issue10-20261017/repo`, on local branch
`issue10-reranker-core-20261017`. Base: setup PR #17 commit
`c97807538f6ab4dea32c350a65c96b8ae20cae4e`. PR #17 was OPEN; main was still
`6677b43858f603c6eebd1190b962d38294ca9822` at the initial check.

Direct GitHub history access on xlogin requested authentication. The explicitly
permitted fallback used existing authorized GitHub access on Fedora to create a
temporary bare clone and Git bundle, then transferred that authentic bundle
through `xlogin login xlogin1`. Fedora temporary history files were removed.
No history was reconstructed. Setup's reference was not modified. The new branch
has no upstream. No repository AGENTS.md or existing Graphify graph was present.

## Environment and exact checks

Python 3.12.3; PyTorch 2.8.0+cpu. Fresh isolated `.venv`, with all unchanged
`requirements.txt` pins installed. No new runtime dependencies or lock changes.
The initial download hit xlogin `/tmp`'s 10 MB user quota. Installation succeeded
with `TMPDIR` pointing to the isolated project's `../tmp`, with no cache, using
the official PyTorch CPU index and package registry. This was a temporary-file
quota issue, not a need to modify network settings or other environments.

From the repository root:

```bash
export TMPDIR=/home/t/tehxa/cs4248-issue10-20261017/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > outputs/environment.txt
.venv/bin/python -m compileall -q reranker tests
.venv/bin/python -m unittest discover -s tests -v
```

- `pip check`: **No broken requirements found.**
- Compilation: passed.
- Final implementation run: **15 tests, 7.166 seconds, OK**.
- Independent review run: **15 tests, 7.066 seconds, OK**.
- Both variants also saved model/config/preprocessing and prediction JSON files;
  reloaded scores were exactly equal. Synthetic expected predictions were
  `{"q1": "right", "q2": "best", "q3": "single"}`.
- The suite includes end-to-end CLI train/save/reload/predict for both variants.

Evidence retained under ignored `outputs/`:

- `reranker-tests.log`, `reranker-independent-review-tests.log`
- `pip-check.txt`, `environment.txt`
- `reranker-synthetic/`: question/target fixtures, feature configurations,
  basic/full model artifacts, basic/full prediction mappings, environment report

## Independent review

A second Codex agent read the implementation/tests/documentation and independently
ran the CPU suite. Two findings were corrected:

1. Public reassignment of fitted feature configuration could silently reinterpret
   learned coefficients. Configuration/hyperparameters now expose read-only
   properties; a regression test checks assignment rejection and stable output.
2. Oversized integer inputs could raise uncaught `OverflowError`. Numeric
   conversion now raises the documented `ValueError`; regressions cover
   hyperparameter, feature, target and model-artifact paths.

The reviewer found no remaining blocking issues after the fixes. Added tests
also check equal per-question loss weighting and that excluded compatibility
and unrelated gold-like fields do not affect fitting. Adapter responsibilities,
all-tied labels versus constant features, and provisional assumptions are
explicit in `reranker.md`.

## Limits and next step

These are synthetic software checks only. No SQuAD/BERT training, actual
validation experiment, final evaluation, or accuracy claim was made. Existing
projects and setup source were preserved. No push, PR, assignment, merge,
teammate message or publication occurred.

Before actual training, settle #9's target definition and no-useful-candidate
policy, its adapter and feature-group contract, and confirm the prediction
mapping with the supplied scorer's owner. Then use a small approved real-data
fixture while preserving training/validation separation. This bounded core does
not close the full training requirement in issue #10.
