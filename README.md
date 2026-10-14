# CS4248 G19 — BERT QA answer reranking

We compare a BERT extractive QA baseline, a lightweight reranker, and a reranker
with question–answer type compatibility features on SQuAD v1.1.

## Setup

Use Python **3.11 or 3.12**. Run commands from the repository root.

```bash
git clone https://github.com/Xateh/ceeyesfortoforate.git
cd ceeyesfortoforate
git switch 1-project-setup  # Until this branch is merged into main.
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install **one** PyTorch build before the other dependencies, with the virtual
environment activated.

**NVIDIA GPU:** check the GPU model and driver with `nvidia-smi`, then choose a
supported build from the
[official PyTorch 2.8 installation options](https://pytorch.org/get-started/previous-versions/#v280).
For example:

- **NVIDIA GeForce RTX 50-series (including RTX 5060 Ti):** use CUDA 12.8
  (`cu128`). These Blackwell GPUs require the newer GPU support; do not use
  `cu126` for them.

  ```bash
  python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  ```

- **Other NVIDIA GPUs / NUS SoC cluster:** `cu126` is another option, **only if
  your GPU and driver support this build**.

  ```bash
  python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
  ```

**CPU only (Linux/Windows):** sufficient for the offline smoke test.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

**macOS:** use `python -m pip install torch==2.8.0`; the smoke test uses CPU.

If you started installing the wrong build, press **Ctrl+C** to cancel. If it
already installed, run `python -m pip uninstall -y torch` before installing the
correct build above. You do not need a separate CUDA Toolkit installation for
these prebuilt wheels, but you do need a compatible NVIDIA driver.

After the chosen PyTorch installation finishes, install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Direct dependencies are pinned. CUDA/CPU builds and transitive packages can
differ between machines; save `python -m pip freeze` with experiment outputs.
Do not install torchvision or torchaudio for this text-only project.

## Verify the environment

First, run the small offline test. It builds a randomly initialised, two-layer
BERT QA model and needs no model download, dataset or GPU:

```bash
python scripts/smoke_test.py --tiny --device cpu --output outputs/smoke-cpu.json
```

Then test the actual generally pretrained BERT checkpoint:

```bash
python scripts/smoke_test.py --device cpu --output outputs/smoke-bert-cpu.json
```

The first full-model run downloads `google-bert/bert-base-uncased` from Hugging
Face (allow several hundred MB and additional RAM for an optimizer step).
The QA head is newly initialised; the corresponding warning is expected.
This check does not train a usable QA system or measure answer quality.

On your own NVIDIA GPU, or **inside an allocated compute session** on a cluster, run:

```bash
nvidia-smi
python scripts/smoke_test.py --device cuda --output outputs/smoke-bert-gpu.json
python -m pip freeze > outputs/environment.txt
```

For NUS SoC, use the cluster's current GPU allocation instructions; do not run
training on a login node. Scheduler, queue and account details must come from
your actual cluster access. This repository does not assume a scheduler.
If compute nodes cannot access the internet, download/cache the model where
network access is allowed and make that Hugging Face cache available to the job.
`HF_HOME` can select a cache directory outside Git.

Success prints `PASS` and saves a JSON report with package versions, model
revision, device/GPU, seed, loss and elapsed time. The script checks finite
loss/gradients, correct logit shapes and updates to both encoder and QA-head
weights. `--device cuda` fails if CUDA is unavailable; it never falls back to CPU.
The tiny test is a software check, not evidence that full BERT works on the GPU.

See [validation notes](docs/setup-validation.md) for what has actually been run.

## Project layout

| Path | Purpose |
| --- | --- |
| `scripts/` | Runnable preprocessing, training and evaluation scripts |
| `configs/` | Small, version-controlled experiment settings |
| `docs/` | Setup and methodology notes |
| `data/` | Local raw/processed data and split manifests (ignored) |
| `checkpoints/` | Local saved model weights (ignored) |
| `outputs/` | Predictions, metrics and environment reports (ignored) |
| `logs/` | Runtime logs (ignored) |

Scripts should create generated directories as needed. Keep data and weights
out of Git; record their download/source locations and settings in documentation.
Commit small configuration files and result summaries needed to reproduce runs.
The data pipeline will split SQuAD training articles into BERT-training,
reranker-training and internal-validation sets. Official dev-v1.1 is reserved
for final evaluation.

## Team workflow

Pick an unassigned issue whose dependencies are merged, then assign yourself
before starting. Usually use one issue, one branch and one PR; include a short
verification command and `Closes #<issue>` in the PR. Review and merge dependent
work before building on it. No one is tied to a particular pipeline stage.

Each member must keep track of their actual AI-tool use for their declaration.
Acknowledge reused code and external resources. The smoke test follows the
[Hugging Face QA model interface](https://huggingface.co/docs/transformers/tasks/question_answering).
