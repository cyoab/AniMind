from __future__ import annotations

import math
from typing import Any


def load_english_eval_texts(
    *,
    dataset_name: str,
    max_samples: int,
    cache_dir: str | None = None,
) -> list[str]:
    if dataset_name != "wikitext2":
        raise RuntimeError(
            "Unsupported English eval dataset. Expected 'wikitext2'. "
            f"Got '{dataset_name}'."
        )
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "English eval dataset requested but datasets package is missing. Install with: uv add datasets"
        ) from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", cache_dir=cache_dir)
    rows: list[str] = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if len(text) < 8:
            continue
        rows.append(text)
        if len(rows) >= max_samples:
            break
    if not rows:
        raise RuntimeError("English eval dataset is empty after filtering.")
    return rows


def compute_english_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_seq_len: int,
    batch_size: int,
    device: str,
    torch_module: Any,
) -> float:
    if not texts:
        raise RuntimeError("Cannot compute English perplexity on empty text set.")
    was_training = bool(model.training)
    model.eval()
    losses: list[float] = []
    with torch_module.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                labels=encoded["input_ids"],
            )
            loss_value = float(outputs.loss.detach().cpu().item())
            if math.isfinite(loss_value):
                losses.append(loss_value)
    if was_training:
        model.train()
    if not losses:
        raise RuntimeError("No finite English eval losses were produced.")
    mean_loss = sum(losses) / len(losses)
    return float(math.exp(min(mean_loss, 20.0)))
