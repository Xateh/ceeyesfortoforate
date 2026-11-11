# Frozen official-dev comparison

From the repository root, with the project dependencies installed, run:

```bash
python -m scripts.final_comparison --data data.zip --candidates dev_candidates.zip
```

Use the data-v1 release archive and Sean's official-dev export. The command checks the two uncompressed input files against the final-run hashes, loads the fitted CPU model in `configs/frozen-basic-model.json`, and prints metrics and paired uncertainty estimates. Add `--predictions-dir outputs/dev-replay` to save both prediction maps to a new directory. The dev candidate ZIP is a separate input; obtain it from the experiment owner. To regenerate it with the original environment, extract data-v1 and bert-finetuned-v1, run the command below and zip the resulting `dev_candidates.json` at the archive root. Hash verification identifies the exact recorded export.

```bash
python scripts/bert_predict.py --checkpoint checkpoints/bert-base-baseline/final --split dev --candidates --batch-size 64 --max-answer-length 30 --output-dir outputs/official-dev
```

The basic ridge reranker uses the five basic features, alpha 1 and training-only preprocessing. It was fitted on 12,516 reranker-training questions. On that same training split, we maximized F1 over observed BERT score gaps, including the option to keep every baseline answer. We retained the smallest threshold in a tie. The frozen rule switches to candidate two only when its reranker score is strictly higher and the difference between the two BERT combined scores is at most 0.02748394012451172. Internal validation informed the experiment design. Official-dev labels were used for scoring after selection.

| Split | System | Exact match (%) | F1 (%) | EM gains / losses vs BERT |
| --- | --- | ---: | ---: | ---: |
| Internal validation | BERT | 65.404 | 79.288 | — |
| Internal validation | Basic, top 20 | 63.925 | 76.782 | 339 / 474 |
| Internal validation | Basic, top 2 | 65.130 | 78.755 | 287 / 312 |
| Internal validation | Frozen rule | 65.404 | 79.300 | 12 / 12 |
| Official dev | BERT | 80.019 | 87.553 | — |
| Official dev | Frozen rule | 80.104 | 87.587 | 18 / 9 |

On internal validation, the top 20 contained an exact answer for 2,461 of BERT's 3,159 errors (77.9%). The unrestricted reranker lost more correct answers than it recovered. Limiting switches to near-ties preserved most baseline answers.

On official dev, the rule changed 68 of 10,570 answers. EM rose by 0.085 percentage points and F1 by 0.034. The two-sided exact McNemar test on the 27 questions whose EM changed gave p = 0.122. For F1, the 95% percentile interval for the paired difference was [−0.018, +0.093] points, resampling the 48 articles with replacement 20,000 times (seed 4248) and weighting questions equally within each resampled dataset. The observed gains are small and the comparison does not establish a reliable improvement.
