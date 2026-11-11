# Code handoff verification: cross-platform reload fix

The packaged model SHA-256 is still checked against the original binding before loading. Inference mutation is checked using before/after snapshots serialized on the same running platform, including the returned score computation. The original frozen models, bindings, binding pins and saved reference predictions are unchanged.

The original bindings also pin the old loader source. A single explicit old-to-corrected loader hash pair is accepted by reranker/reload_compatibility.py. Model hashes and every other source hash remain strict; arbitrary source changes are rejected. provenance/reload-platform-fix.json records this transition and the preserved artifacts.

## Checks actually run

- **Windows 11, Python 3.12.10, CPU torch 2.8.0:** all 143 tests across 13 suites completed: 142 passed and one skipped. The skip is only test_exclusive_save_preserves_symlink_target because Windows reports WinError 1314 when this account creates a symlink. Ordinary file-preservation tests pass. All six main prediction maps exactly match the saved references.
- **Ubuntu under WSL2, Python 3.12.3, CPU torch 2.8.0:** 68 tests passed with zero skips across test_submission_inference, test_real_training, test_reranker, test_team_model, test_team_candidates and test_evaluator_boundary. The CRLF regression executed natively with LF serialization; altered model bytes and in-memory predict/scores mutations were rejected. All six main prediction maps exactly match the saved references. The contextual/transformers suites were not run on Linux.
- Both main replays checked all 9,131 internal and 10,570 official-dev IDs, the same EM/F1 table, fixes/breaks and course/repository per-question scorer parity. No BERT training, GPU inference or new experiments were performed. Regression tests fit only tiny synthetic fixtures. The older full experiment replay is retained as dated historical evidence, not represented as newly rerun.
- No macOS execution was performed.

## Evidence supplied with this handoff

The rebuilt code ZIP includes verification/summary.json and verification/platform-fix/ with the Windows/Linux test logs, replay receipts and environment versions. Previous handoff evidence is retained under verification/previous-handoff/ and explicitly predates this loader change.

Final tests of the rebuilt ZIP itself are recorded separately in **verification/archive-replay/** alongside the three ZIPs: Windows and Linux preparation/replay logs, receipts, archive checksums and a summary. This evidence is generated after the code ZIP is sealed, so it is supplied as a sidecar folder rather than silently changing that ZIP after testing. Copy the entire verification folder with the ZIPs for team review. No original-PC path is needed to read those relative evidence files.

SHA256SUMS.txt beside the ZIPs is the authoritative archive checksum list. The code ZIP has a rebuilt provenance/package-files.json; every listed internal payload hash was checked after extraction. The data and checkpoint companions retain their previous exact bytes. PR #31 remains open; no release or Canvas submission is included.

The historical report bootstrap discrepancy remains: the internal-validation gap-plus-span calibrated contextual experiment has documented 95% F1-delta interval [-0.120, +0.219], while its report snapshot says [-0.119, +0.221]. This is unrelated to the loader fix and does not change any of the six main prediction maps or scores.
