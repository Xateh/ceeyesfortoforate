# Results and issue #11–13 evidence

| Split | System | EM | F1 | Fixes / breaks |
|---|---|---:|---:|---:|
| Internal validation (9,131) | bert | 65.403570 | 79.287794 | — |
| Internal validation (9,131) | basic | 63.925090 | 76.781898 | 339 / 474 |
| Internal validation (9,131) | full | 63.892235 | 76.772235 | 337 / 475 |
| Official dev (10,570) | bert | 80.018921 | 87.553216 | — |
| Official dev (10,570) | basic | 78.505203 | 85.792512 | 399 / 559 |
| Official dev (10,570) | full | 78.533586 | 85.830888 | 400 / 557 |

Neither all-candidate reranker improves on BERT. Full versus basic differs by −0.032855 EM / −0.009662 F1 on internal validation and +0.028382 EM / +0.038376 F1 on official dev. These small differences do not establish a reliable benefit from the two compatibility interactions. The eight question-type indicators cancel within each question.

## Internal validation: question-type breakdown

| Type | n | Basic EM | Full EM | Basic F1 | Full F1 | Basic fixes/breaks | Full fixes/breaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| who | 753 | 77.025 | 76.760 | 85.108 | 85.067 | 14/29 | 12/29 |
| what | 3990 | 60.777 | 60.727 | 74.453 | 74.434 | 166/246 | 165/247 |
| when | 593 | 80.607 | 80.776 | 88.005 | 88.174 | 19/16 | 20/16 |
| where | 377 | 62.334 | 62.334 | 74.663 | 74.663 | 15/17 | 15/17 |
| why | 126 | 19.841 | 19.841 | 53.057 | 53.057 | 8/15 | 8/15 |
| how | 854 | 56.909 | 56.792 | 74.093 | 73.971 | 27/37 | 27/38 |
| which | 340 | 63.824 | 63.824 | 76.255 | 76.255 | 13/15 | 13/15 |
| other | 2098 | 66.301 | 66.349 | 78.037 | 78.046 | 77/99 | 77/98 |

Real examples (both linear models select these same answers):

- **fix:** When did Justinian's reign end? Reference: '565'; BERT: '527–565'; reranked: '565'. ID `56fb3441f34c681400b0c20d`.
- **break:** Who was Allan Loren? Reference: 'Apple USA head'; BERT: 'Apple USA head'; reranked: 'head'. ID `570d2954fed7b91900d45c82`.

## Official dev: question-type breakdown

| Type | n | Basic EM | Full EM | Basic F1 | Full F1 | Basic fixes/breaks | Full fixes/breaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| who | 1062 | 86.911 | 87.006 | 90.453 | 90.497 | 23/33 | 23/32 |
| what | 4766 | 76.018 | 76.018 | 84.455 | 84.455 | 208/305 | 208/305 |
| when | 696 | 88.793 | 89.080 | 92.639 | 93.058 | 14/22 | 15/21 |
| where | 433 | 72.055 | 72.055 | 81.367 | 81.367 | 19/28 | 19/28 |
| why | 151 | 39.073 | 39.073 | 62.940 | 62.940 | 8/34 | 8/34 |
| how | 1090 | 78.899 | 78.899 | 86.206 | 86.206 | 49/43 | 49/43 |
| which | 455 | 76.264 | 76.264 | 82.952 | 82.952 | 16/25 | 16/25 |
| other | 1917 | 81.168 | 81.168 | 87.289 | 87.324 | 62/69 | 62/69 |

Real examples (both linear models select these same answers):

- **fix:** When did Zhu Shijie die? Reference: '1314'; BERT: '1249–1314'; reranked: '1314'. ID `5728804b4b864d1900164a47`.
- **break:** Where was Luther born? Reference: 'Eisleben, Saxony'; BERT: 'Eisleben, Saxony'; reranked: 'Saxony'. ID `56f7cdc2a6d7ea1400e1728f`.

## Oracle, ablation and contextual experiments

The internal top-20 oracle scores 92.355711 EM / 95.722708 F1. It contains an exact answer for 2,461 of BERT's 3,159 errors (77.9044%); 698 errors cannot be recovered with any candidate. This is a gold-aware diagnostic, not deployable inference.

| Internal-validation experiment | EM | F1 | Fixes / breaks |
|---|---:|---:|---:|
| Basic without span length | 65.184536 | 79.179459 | 53 / 73 |
| No-length plus list feature (exploratory, not report row) | 65.053116 | 79.079276 | 57 / 89 |
| CLS epoch 5 | 65.403570 | 79.287794 | 0 / 0 |
| Span-mean epoch 5 | 65.414522 | 79.295613 | 1 / 0 |
| Gap-only calibration | 65.381667 | 79.285790 | 26 / 28 |
| Gap + span-score calibration | 65.436425 | 79.339035 | 81 / 78 |

CLS is rescored from its original epoch-5 predictions, exactly equal to BERT. Span-mean and both calibrations are replayed from retained CSV scores on CPU, with train-only calibration fitting. The uncalibrated span-mean predictions match the original epoch-5 file. The two CSVs are included in the code ZIP under inputs/context-scores. Original GPU fitting is not rerun.

The separate near-tie gated top-two experiment scores official dev 80.104068 EM / 87.587468 F1, with 18 fixes and 9 breaks over 68 answer changes. It is not the basic top-20 model in the main table. See results/gated-top-two and docs/final-comparison.md.

Issue #11 evidence: frozen basic/full models, one-alpha policy, verified internal results, ablation and type table. Issue #12 evidence: exact official ID sets, predictions saved before gold, unchanged course CLI scores and hashes. Issue #13 evidence: oracle, corrections/regressions, type counts and real examples. The issues themselves remain unchanged.
