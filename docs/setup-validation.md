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

## Remaining check for issue #1

This environment has no NVIDIA GPU or authenticated NUS SoC cluster session.
The intended-GPU/cluster smoke test is **pending**. In an allocated GPU session,
follow the README installation steps and run:

```bash
nvidia-smi
python scripts/smoke_test.py --device cuda --output outputs/smoke-bert-gpu.json
python -m pip freeze > outputs/environment.txt
```

Record the GPU, cluster allocation command (without credentials), PyTorch/CUDA
versions and successful result in the PR before closing issue #1. Keep detailed
generated reports under `outputs/`; commit only a small summary here if useful.
