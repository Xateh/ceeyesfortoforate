# Setup validation

Validated on 14 October 2026 in a fresh Linux virtual environment with Python
3.12.14, PyTorch 2.8.0+cpu, Transformers 4.57.1, Datasets 4.3.0 and Accelerate
1.10.1. The filesystem backend is pinned to fsspec 2025.9.0, compatible with
Datasets 4.3.0. Python 3.11 and Windows/macOS instructions have not been run here.

| Check | Result |
| --- | --- |
| `python -m pip check` | Passed: no broken requirements |
| Datasets/Accelerate imports and a one-row in-memory dataset | Passed |
| `python scripts/smoke_test.py --tiny --device cpu --output outputs/smoke-cpu.json` | Passed; loss 2.780924; logits `[1, 16]` |
| `python scripts/smoke_test.py --device cpu --output outputs/smoke-bert-cpu.json` | Passed; loss 2.834783; logits `[1, 17]` |
| `python scripts/smoke_test.py --tiny --device cuda --output outputs/should-not-exist.json` | Expected exit 2: CUDA unavailable; no success report written |

Both successful runs verified finite loss/gradients and optimizer updates to
the BERT encoder and QA head. The full-model run loaded
`google-bert/bert-base-uncased` at revision
`86b5e0934494bd15c9632b12f734a8a67f723594`, with a newly initialised QA head.
These are environment checks using one artificial example, not QA results.

## Sean's local CPU and GPU validation

Sean supplied successful smoke-test JSON reports on 14 October 2026 from his
local PC, using Python 3.12.10, PyTorch 2.8.0+cu128, CUDA runtime 12.8,
Transformers 4.57.1, Datasets 4.3.0 and Accelerate 1.10.1.

| Run | Device | Loss | Elapsed seconds | Result |
| --- | --- | --- | --- | --- |
| Tiny offline BERT | CPU | 2.780924 | 0.401 | Passed |
| Pretrained BERT | CPU | 2.834783 | 165.786 | Passed |
| Pretrained BERT | NVIDIA GeForce RTX 5060 Ti (CUDA) | 3.014746 | 1.801 | Passed |

The GPU report was generated at 2026-10-14 12:16:54 UTC. Both full-model runs
used the checkpoint revision recorded above and produced logits of shape
`[1, 17]`. All three reports confirm finite loss/gradients and updates to both
the encoder and QA head. Timings include setup/model loading and are not
training-throughput benchmarks.

The intended local GPU check for issue #1 is complete. NUS SoC cluster execution
has not been tested; validate there separately if the team later uses it.
Keep the original generated reports under local `outputs/`.
