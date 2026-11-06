# Reranker report contribution — draft for review, 6 November 2026

The proposed reranker is a linear float64 CPU scorer trained with equal weight
per informative question and within-question-centered ridge loss. Feature scaling
is fitted on reranker-training candidates only. Single-candidate and equal-target
questions remain in coverage/scaling but contribute no ranking loss; a wholly
uninformative batch fails. Exact score ties preserve the supplied candidate order.

Basic uses start/end logits, token count, ASCII digit presence and initial ASCII
uppercase. Full adds eight fixed question-type indicators and two interactions;
basic excludes all question-type features. This registry and the recorded target,
alpha-grid and no-signal policies are explicit experiment proposals, not claimed
team agreements. Targets use F1 from the pinned repository evaluator, separate
from input features. Both variants support internal-validation selection,
baseline comparison, saved model/configuration/provenance and checked reload.

The intended inputs are Hai's Canvas-based seed-4248 75/15/10 splits and checkpoint,
plus Sean's matching schema-1 candidate exports and independent internal-validation
baseline. BERT optimizer seed 42 is distinct from the split seed. The data release ZIP matches its approved size and SHA-256. Direct inspection of
its split JSONs confirmed 332/66/44 articles and 65,952/12,516/9,131 questions,
with no detected duplicate IDs/titles, cross-role exact-context/article/question
overlaps or invalid answer spans. A separate archive-only canonical comparison verified the complete, disjoint
source union and exact article/question/context/answer identities against the
archive’s original training JSON. Independent Canvas authenticity and seeded
allocation reproduction remain unverified. No official-dev golds were read. The checkpoint
ZIP was completed by one approved append-only resume and matches its approved
whole-file hash; the original prefix was preserved. It contains model weights
and tokenizer/config files. Inspected metadata declares BERT QA and lowercase
WordPiece with matching 30,522-entry vocabulary sizes. No weights were loaded.
Training run-info/log/environment records are absent from the archive; exact
base revision, training/export correspondence, real candidate exports and original
course-scorer provenance remain pending. Repository evaluator bytes were exercised on invented fixtures;
this does not authenticate the original course download. Main's actual baseline
compares all valid windows without the max-context filter stated in older docs.

This real-label increment passed 35 selected fixture/regression tests: eight new
integration tests, 18 core regressions and nine synthetic-model persistence
regressions. These are distinct from the earlier 94 synthetic regressions at
consolidation commit 619f77fe3cd12f02c2f036da19b94bd476524519. These are software
checks, not SQuAD accuracy. No real-data fitting, BERT inference, new real EM/F1,
coverage/oracle analysis or final-dev comparison was performed. Published baseline
metrics are team results, not this task's rerun; official dev is not untouched.

Before real execution, verify release/split/export/model/tokenizer/scorer hashes
and lineage, record the accepted run policy and resource allowance, then retain
predictions, metrics, selected alpha values, environment/runtime receipts and
independently trusted model-binding pins. Hashes identify bytes, not origin.
This contribution is provided for PR review; coursework submission remains pending.

OpenAI Codex generated the runner, tests and this draft, with an independent
Codex source review. Xu An must review and accurately disclose this assistance.
This draft does not claim personal authorship of AI-written text or experiments
that were not performed.
