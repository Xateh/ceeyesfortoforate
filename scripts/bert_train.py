"""Fine-tune pretrained BERT and QA head on bert_train.

Usage:
    python scripts/train_bert.py --smoke     # quick check: a few steps on a small subset
    python scripts/train_bert.py             # full run
    python scripts/train_bert.py --resume    # continue a full run from its latest checkpoint

Outputs, in --output-dir (default checkpoints/<run_name>, or <run_name>-smoke):
    checkpoint-<step>/   periodic checkpoints (for --resume)
    final/               final model + tokenizer, loadable with from_pretrained()
    run_info.json        config, device, precision, package versions, metrics
    train_log.jsonl      one line per logging step (loss, learning rate, ...)
    train.log            the script's messages
    environment.txt      pip freeze
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments, default_data_collator, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import METADATA_COLUMNS, load_examples, span_text

MODEL_INPUTS = ("input_ids", "token_type_ids", "attention_mask")
log = logging.getLogger("train_bert")


def load_tokenized(path):
    path = Path(path)
    if path.is_dir():
        return load_from_disk(str(path))
    parquet = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
    if parquet.exists():
        return Dataset.from_parquet(str(parquet))
    raise FileNotFoundError(f"No tokenised data at {path}. Run scripts/preprocessing.py first.")

def choose_precision(requested, device):
    if device.type != "cuda":
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but this GPU does not support it; use fp16 or auto.")
    return requested

class JsonlLogger(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}) + "\n")
            log.info("step %d: %s", state.global_step, logs)

def best_span(start_logits, end_logits, offsets, n_best=20, max_answer_tokens=30):
    best, best_score = None, -np.inf
    for s in np.argsort(start_logits)[::-1][:n_best]:
        for e in np.argsort(end_logits)[::-1][:n_best]:
            if offsets[s] is None or offsets[e] is None or e < s or e - s + 1 > max_answer_tokens:
                continue
            score = start_logits[s] + end_logits[e]
            if score > best_score:
                best, best_score = (int(s), int(e)), score
    return best

def check_reload(final_dir, eval_features, examples_path, device, n_windows=32):
    # load saved model and perform some test predictions
    tokenizer = AutoTokenizer.from_pretrained(final_dir) 
    model = AutoModelForQuestionAnswering.from_pretrained(final_dir).to(device).eval()
    rows = eval_features.select(range(min(n_windows, len(eval_features))))
    batch = {key: torch.tensor(rows[key], dtype=torch.long, device=device) for key in MODEL_INPUTS}
    with torch.no_grad():
        output = model(**batch)
    start_logits = output.start_logits.float().cpu().numpy()
    end_logits = output.end_logits.float().cpu().numpy()
    if start_logits.shape != tuple(batch["input_ids"].shape) or not np.isfinite(start_logits).all():
        raise RuntimeError("Reloaded model produced unexpected or non-finite logits.")

    examples = {example["id"]: example for example in load_examples(examples_path)}
    predictions, exact = [], 0
    for i, row in enumerate(rows):
        example = examples[row["example_id"]]
        span = best_span(start_logits[i], end_logits[i], row["offset_mapping"])
        text = span_text(example["context"], row["offset_mapping"], *span) if span else ""
        exact += text in example["answers"]["text"]
        predictions.append({"question": example["question"], "predicted": text,
                            "gold": example["answers"]["text"][0]})
    log.info("Reloaded %s (tokenizer: %s) and predicted %d windows; %d exactly match a gold "
             "answer (sanity check only, not an evaluation).",
             final_dir, type(tokenizer).__name__, len(rows), exact)
    for p in predictions[:3]:
        log.info("  Q: %s | predicted: %r | gold: %r", p["question"], p["predicted"], p["gold"])
    return {"windows": len(rows), "exact_match_windows": exact, "examples": predictions[:3]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/bert-train.json"))
    parser.add_argument("--smoke", action="store_true", help="Train a few steps on a small subset.")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint.")
    parser.add_argument("--output-dir", type=Path, help="Where to write checkpoints and logs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = {key: value for key, value in config.items() if key != "smoke"}
    if args.smoke:
        settings.update(config["smoke"])
    run_dir = args.output_dir or Path("checkpoints") / (config["run_name"] + ("-smoke" if args.smoke else ""))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable.")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    precision = choose_precision(settings["precision"], device)
    if settings["effective_batch_size"] % settings["per_device_batch_size"]:
        parser.error("effective_batch_size must be a multiple of per_device_batch_size.")
    accumulation = settings["effective_batch_size"] // settings["per_device_batch_size"]

    # only resume training if same settings
    info_path = run_dir / "run_info.json"
    has_checkpoints = run_dir.exists() and any(run_dir.glob("checkpoint-*"))
    if args.resume:
        if not has_checkpoints:
            parser.error(f"--resume given but {run_dir} has no checkpoints.")
        if json.loads(info_path.read_text(encoding="utf-8"))["settings"] != settings:
            parser.error(f"Settings differ from the run in {run_dir}; use a new --output-dir.")
    elif has_checkpoints and not args.smoke:
        parser.error(f"{run_dir} already has checkpoints; use --resume or a new --output-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(run_dir / "train.log", encoding="utf-8")])
    set_seed(settings["seed"])

    train = load_tokenized(settings["train_data"]).remove_columns(list(METADATA_COLUMNS))
    eval_features = load_tokenized(settings["eval_data"])  # keeps metadata for the reload check
    if args.smoke:
        train = train.select(range(min(settings["train_windows"], len(train))))
        eval_features = eval_features.select(range(min(settings["eval_windows"], len(eval_features))))
    eval_inputs = eval_features.remove_columns(list(METADATA_COLUMNS))

    tokenizer = AutoTokenizer.from_pretrained(settings["model"], use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(settings["model"])  # QA head newly initialised
    if model.config.model_type != "bert":
        raise ValueError("Expected a BERT checkpoint.")

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        seed=settings["seed"],
        data_seed=settings["seed"],
        learning_rate=settings["learning_rate"],
        num_train_epochs=settings["num_train_epochs"],
        max_steps=settings.get("max_steps", -1),
        per_device_train_batch_size=settings["per_device_batch_size"],
        per_device_eval_batch_size=settings["per_device_batch_size"] * 2,
        gradient_accumulation_steps=accumulation,
        warmup_ratio=settings["warmup_ratio"],
        weight_decay=settings["weight_decay"],
        adam_epsilon=settings["adam_epsilon"],
        lr_scheduler_type=settings["lr_scheduler"],
        max_grad_norm=settings["max_grad_norm"],
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        logging_steps=settings["logging_steps"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        eval_strategy="no" if args.smoke else "epoch",  # eval loss on internal_val, never dev
        dataloader_num_workers=settings["dataloader_num_workers"],
        use_cpu=device.type == "cpu",
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=eval_inputs,
                      data_collator=default_data_collator, processing_class=tokenizer,
                      callbacks=[JsonlLogger(run_dir / "train_log.jsonl")])

    info = {
        "status": "running",
        "smoke": args.smoke,
        "resumed": args.resume,
        "settings": settings,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision,
        "gradient_accumulation_steps": accumulation,
        "train_windows": len(train),
        "eval_windows": len(eval_inputs),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "datasets", "accelerate")},
        "cuda_runtime": torch.version.cuda,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, check=True).stdout
        (run_dir / "environment.txt").write_text(freeze, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError):
        log.warning("Could not run pip freeze; environment.txt not written.")
    log.info("Training on %s (%s, %s): %d windows, effective batch %d (%d x %d accumulation)",
             device, info["gpu"] or "no GPU", precision, len(train), settings["effective_batch_size"],
             settings["per_device_batch_size"], accumulation)

    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))  # model + tokenizer
    eval_metrics = trainer.evaluate()
    # Last logged training loss (train_result.training_loss is misleading after --resume).
    last_loss = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry][-1]
    if not np.isfinite(last_loss):
        raise RuntimeError("Training loss is not finite.")

    info.update({
        "status": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "reload_check": check_reload(final_dir, eval_features, settings["eval_examples"], device),
    })
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    log.info("PASS: trained %d steps, last train loss %.4f, internal_val loss %.4f; saved %s",
             train_result.global_step, last_loss, eval_metrics["eval_loss"], final_dir)


if __name__ == "__main__":
    main()