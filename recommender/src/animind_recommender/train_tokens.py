from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from .train_types import TokenExtensionConfig

_SID_PATTERN = re.compile(r"^<L(\d+)_(\d+)>$")


@dataclass(slots=True)
class TokenExtensionResult:
    original_vocab_size: int
    final_vocab_size: int
    added_tokens: int
    added_special_tokens: int
    warm_start_applied_tokens: int
    warm_start_fit_rows: int
    warm_start_fit_mse: float | None


def extend_tokenizer_and_embeddings(
    *,
    tokenizer: Any,
    model: Any,
    cfg: TokenExtensionConfig,
    console: Console,
) -> TokenExtensionResult:
    if not cfg.semantic_vocab_path.exists():
        raise RuntimeError(f"Semantic vocab file not found: {cfg.semantic_vocab_path}")
    vocab_payload = json.loads(cfg.semantic_vocab_path.read_text(encoding="utf-8"))
    vocab_tokens = [str(token) for token in vocab_payload.get("tokens", [])]
    special_tokens = [str(token) for token in vocab_payload.get("special_tokens", [])]
    if not vocab_tokens:
        raise RuntimeError(f"No tokens found in semantic vocab file: {cfg.semantic_vocab_path}")

    original_vocab_size = len(tokenizer)
    added_special_tokens = 0
    if cfg.add_special_tokens and special_tokens:
        added_special_tokens = int(tokenizer.add_special_tokens({"additional_special_tokens": special_tokens}))
    added_regular_tokens = int(tokenizer.add_tokens(vocab_tokens, special_tokens=False))
    final_vocab_size = len(tokenizer)

    if added_special_tokens > 0 or added_regular_tokens > 0:
        model.resize_token_embeddings(final_vocab_size)
        console.log(
            "Tokenizer extension complete: "
            f"original_vocab={original_vocab_size:,}, "
            f"added_special={added_special_tokens:,}, "
            f"added_regular={added_regular_tokens:,}, "
            f"final_vocab={final_vocab_size:,}."
        )
    else:
        console.log(
            "Tokenizer extension found existing tokens only; "
            f"vocab unchanged at {final_vocab_size:,}."
        )

    warm_tokens = 0
    warm_fit_rows = 0
    warm_mse: float | None = None
    if cfg.warm_start.enabled:
        warm_tokens, warm_fit_rows, warm_mse = apply_warm_start(
            tokenizer=tokenizer,
            model=model,
            warm_cfg=cfg.warm_start,
            base_vocab_size=original_vocab_size,
            console=console,
        )

    return TokenExtensionResult(
        original_vocab_size=original_vocab_size,
        final_vocab_size=final_vocab_size,
        added_tokens=int(added_regular_tokens),
        added_special_tokens=int(added_special_tokens),
        warm_start_applied_tokens=warm_tokens,
        warm_start_fit_rows=warm_fit_rows,
        warm_start_fit_mse=warm_mse,
    )


def apply_warm_start(
    *,
    tokenizer: Any,
    model: Any,
    warm_cfg: Any,
    base_vocab_size: int,
    console: Console,
) -> tuple[int, int, float | None]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Warm-start requested but torch is not installed. Install with: uv add torch"
        ) from exc

    if not warm_cfg.semantic_ids_path.exists():
        raise RuntimeError(
            "Warm-start requested but semantic_ids file not found: "
            f"{warm_cfg.semantic_ids_path}"
        )
    if not warm_cfg.rqvae_checkpoint_path.exists():
        raise RuntimeError(
            "Warm-start requested but RQ-VAE checkpoint not found: "
            f"{warm_cfg.rqvae_checkpoint_path}"
        )

    checkpoint = torch.load(warm_cfg.rqvae_checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Warm-start checkpoint missing model_state_dict.")
    if "quantizer.codebook" not in state_dict:
        raise RuntimeError("Warm-start checkpoint missing quantizer.codebook.")
    codebook = state_dict["quantizer.codebook"].detach().cpu().to(torch.float32)
    if int(codebook.dim()) != 2:
        raise RuntimeError("Warm-start codebook has invalid shape.")

    embeddings = model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    hidden_size = int(embeddings.size(1))
    latent_dim = int(codebook.size(1))
    console.log(
        "Warm-start fit setup: "
        f"codebook_size={codebook.size(0):,}, latent_dim={latent_dim}, hidden_size={hidden_size}."
    )

    fit_rows = _build_warm_fit_rows(
        semantic_ids_path=warm_cfg.semantic_ids_path,
        tokenizer=tokenizer,
        embeddings=embeddings,
        codebook=codebook,
        base_vocab_size=base_vocab_size,
        max_fit_samples=int(warm_cfg.max_fit_samples),
    )
    if fit_rows is None:
        raise RuntimeError("Warm-start failed: no usable fit rows could be built.")
    source_matrix, target_matrix = fit_rows
    fit_count = int(source_matrix.size(0))
    if fit_count < 32:
        raise RuntimeError(f"Warm-start fit rows are insufficient: {fit_count}")

    lambda_value = float(warm_cfg.ridge_lambda)
    eye = torch.eye(latent_dim, dtype=torch.float32)
    xtx = source_matrix.T @ source_matrix
    xty = source_matrix.T @ target_matrix
    weights = torch.linalg.solve(xtx + (lambda_value * eye), xty)
    predictions = source_matrix @ weights
    mse = float(torch.mean((predictions - target_matrix) ** 2).item())
    if not math.isfinite(mse):
        raise RuntimeError("Warm-start fit produced non-finite MSE.")

    sid_tokens = _extract_sid_tokens(tokenizer=tokenizer)
    applied = 0
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        for token in sid_tokens:
            parsed = _parse_sid_token(token)
            if parsed is None:
                continue
            _level, code = parsed
            if code < 0 or code >= int(codebook.size(0)):
                continue
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            if token_id < 0:
                continue
            projected = (codebook[code].unsqueeze(0) @ weights).squeeze(0).to(emb_layer.weight.device)
            emb_layer.weight[token_id] = projected.to(emb_layer.weight.dtype)
            applied += 1

    console.log(
        "[green]Warm-start applied.[/green] "
        f"sid_tokens_updated={applied:,}, fit_rows={fit_count:,}, fit_mse={mse:.6f}."
    )
    return applied, fit_count, mse


def _build_warm_fit_rows(
    *,
    semantic_ids_path: Path,
    tokenizer: Any,
    embeddings: Any,
    codebook: Any,
    base_vocab_size: int,
    max_fit_samples: int,
) -> tuple[Any, Any] | None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Warm-start requested but torch is not installed. Install with: uv add torch"
        ) from exc

    source_rows: list[Any] = []
    target_rows: list[Any] = []
    with semantic_ids_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            title = str(row.get("english_name") or row.get("name") or "").strip()
            if not title:
                continue
            codes = [int(v) for v in row.get("codes", []) if isinstance(v, int) or str(v).isdigit()]
            if not codes:
                continue
            if any(code < 0 or code >= int(codebook.size(0)) for code in codes):
                continue
            token_ids = list(tokenizer(title, add_special_tokens=False).input_ids)
            token_ids = [token_id for token_id in token_ids if int(token_id) < base_vocab_size]
            if not token_ids:
                continue

            source = codebook[codes].mean(dim=0)
            target = embeddings[token_ids].mean(dim=0)
            source_rows.append(source)
            target_rows.append(target)
            if max_fit_samples > 0 and len(source_rows) >= max_fit_samples:
                break
    if not source_rows:
        return None
    return torch.stack(source_rows, dim=0), torch.stack(target_rows, dim=0)


def _extract_sid_tokens(*, tokenizer: Any) -> list[str]:
    vocab = tokenizer.get_vocab()
    return [token for token in vocab.keys() if _SID_PATTERN.fullmatch(str(token))]


def _parse_sid_token(token: str) -> tuple[int, int] | None:
    match = _SID_PATTERN.fullmatch(token)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))
