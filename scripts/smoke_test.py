"""Check one BERT QA forward/backward/optimizer step; no SQuAD data needed."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BertConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tiny", action="store_true", help="Use a small random BERT offline.")
    parser.add_argument("--model", default="google-bert/bert-base-uncased",
                        help="Generally pretrained BERT checkpoint or local directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke-test.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable. Check your GPU allocation and PyTorch build.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    # Avoid excessive CPU thread overhead for this one-example check.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = time.perf_counter()

    if args.tiny:
        config = BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                            num_attention_heads=4, intermediate_size=64,
                            max_position_embeddings=32)
        model = AutoModelForQuestionAnswering.from_config(config)
        batch = {
            "input_ids": torch.randint(1, 128, (1, 16)),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
            "token_type_ids": torch.tensor([[0] * 8 + [1] * 8]),
            "start_positions": torch.tensor([10]),
            "end_positions": torch.tensor([11]),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
        if model.config.model_type != "bert":
            raise ValueError("This project's smoke test expects a BERT checkpoint.")
        question = "What is the capital of France?"
        context = "The capital of France is Paris."
        encoded = tokenizer(question, context, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        answer_start = context.index("Paris")
        answer_end = answer_start + len("Paris")
        answer_tokens = [i for i, (start, end) in enumerate(offsets)
                         if encoded.sequence_ids(0)[i] == 1
                         and start < answer_end and end > answer_start]
        if not answer_tokens:
            raise RuntimeError("Smoke-test answer did not align with the tokenised context.")
        batch = dict(encoded)
        batch["start_positions"] = torch.tensor([answer_tokens[0]])
        batch["end_positions"] = torch.tensor([answer_tokens[-1]])

    model.to(device).train()
    batch = {key: value.to(device) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    # Check that both the encoder and QA head participate in learning.
    watched = {"encoder": model.bert.embeddings.word_embeddings.weight,
               "qa_head": model.qa_outputs.weight}
    before = {name: parameter.detach().clone() for name, parameter in watched.items()}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    expected_shape = batch["input_ids"].shape
    if outputs.start_logits.shape != expected_shape or outputs.end_logits.shape != expected_shape:
        raise RuntimeError("Unexpected start/end logit shapes.")
    if outputs.loss is None or not torch.isfinite(outputs.loss).item():
        raise RuntimeError("Expected a finite QA loss.")
    outputs.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Non-finite gradient: {name}")
    for name, parameter in watched.items():
        if parameter.grad is None or not torch.count_nonzero(parameter.grad).item():
            raise RuntimeError(f"No nonzero gradient reached the {name}.")
    optimizer.step()
    for name, parameter in watched.items():
        if not torch.isfinite(parameter).all().item() or torch.equal(before[name], parameter.detach()):
            raise RuntimeError(f"The {name} did not receive a finite parameter update.")
    if device.type == "cuda":
        torch.cuda.synchronize()

    report = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "tiny-random-offline" if args.tiny else "pretrained-bert",
        "model": "tiny-random-bert" if args.tiny else args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "transformers", "datasets", "accelerate")},
        "cuda_runtime": torch.version.cuda,
        "loss": outputs.loss.item(),
        "logit_shape": list(outputs.start_logits.shape),
        "encoder_and_qa_head_updated": True,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"PASS: report saved to {args.output}")


if __name__ == "__main__":
    main()
