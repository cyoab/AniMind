from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .embed import EmbedConfig
from .prep import PrepConfig
from .rqvae import RQVAEConfig


def build_prep_config(config_path: Path) -> PrepConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="prep")

    source_db = _resolve_path(config_path=config_path, raw_path=section.get("source_db", "../output/anilist.sqlite"))
    out_dir = _resolve_path(config_path=config_path, raw_path=section.get("out_dir", "./output"))
    rebuild = bool(section.get("rebuild", True))
    export_parquet = bool(section.get("export_parquet", True))
    limit = int(section.get("limit", 0))
    if limit < 0:
        raise RuntimeError("Invalid prep.limit in config: must be >= 0")

    return PrepConfig(
        source_db=source_db,
        out_dir=out_dir,
        rebuild=rebuild,
        export_parquet=export_parquet,
        limit=limit,
    )


def build_embed_config(config_path: Path) -> EmbedConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="embedd")

    tokenizer_db = _resolve_path(
        config_path=config_path,
        raw_path=section.get("tokenizer_db", "./output/tokenizer.sqlite"),
    )
    rebuild = bool(section.get("rebuild", True))
    limit = int(section.get("limit", 0))
    if limit < 0:
        raise RuntimeError("Invalid embedd.limit in config: must be >= 0")

    batch_size = int(section.get("batch_size", 8))
    max_length = int(section.get("max_length", 2048))
    if batch_size < 1:
        raise RuntimeError("Invalid embedd.batch_size in config: must be >= 1")
    if max_length < 8:
        raise RuntimeError("Invalid embedd.max_length in config: must be >= 8")

    return EmbedConfig(
        tokenizer_db=tokenizer_db,
        rebuild=rebuild,
        limit=limit,
        model_name=str(section.get("model_name", "tencent/KaLM-Embedding-Gemma3-12B-2511")),
        batch_size=batch_size,
        max_length=max_length,
        device=str(section.get("device", "auto")),
        normalize=bool(section.get("normalize", True)),
    )


def build_rqvae_config(config_path: Path) -> RQVAEConfig:
    doc = _load_toml(config_path=config_path)
    section = _require_section(doc=doc, section_name="rqvae")

    tokenizer_db = _resolve_path(
        config_path=config_path,
        raw_path=section.get("tokenizer_db", "./output/tokenizer.sqlite"),
    )
    out_dir = _resolve_path(
        config_path=config_path,
        raw_path=section.get("out_dir", "./output/rqvae"),
    )
    env_file = _resolve_path(
        config_path=config_path,
        raw_path=section.get("env_file", "./.env"),
    )
    rebuild = bool(section.get("rebuild", True))
    limit = int(section.get("limit", 0))
    if limit < 0:
        raise RuntimeError("Invalid rqvae.limit in config: must be >= 0")

    batch_size = int(section.get("batch_size", 256))
    epochs = int(section.get("epochs", 40))
    num_workers = int(section.get("num_workers", 4))
    if batch_size < 1:
        raise RuntimeError("Invalid rqvae.batch_size in config: must be >= 1")
    if epochs < 1:
        raise RuntimeError("Invalid rqvae.epochs in config: must be >= 1")
    if num_workers < 0:
        raise RuntimeError("Invalid rqvae.num_workers in config: must be >= 0")

    val_ratio = float(section.get("val_ratio", 0.05))
    if val_ratio <= 0 or val_ratio >= 1:
        raise RuntimeError("Invalid rqvae.val_ratio in config: must be > 0 and < 1")

    lr = float(section.get("lr", 0.00004))
    warmup_steps = int(section.get("warmup_steps", 1000))
    if lr <= 0:
        raise RuntimeError("Invalid rqvae.lr in config: must be > 0")
    if warmup_steps < 0:
        raise RuntimeError("Invalid rqvae.warmup_steps in config: must be >= 0")

    latent_dim = int(section.get("latent_dim", 256))
    rq_levels = int(section.get("rq_levels", 8))
    codebook_size = int(section.get("codebook_size", 2048))
    if latent_dim < 8:
        raise RuntimeError("Invalid rqvae.latent_dim in config: must be >= 8")
    if rq_levels < 1:
        raise RuntimeError("Invalid rqvae.rq_levels in config: must be >= 1")
    if codebook_size < 32:
        raise RuntimeError("Invalid rqvae.codebook_size in config: must be >= 32")

    commitment_beta = float(section.get("commitment_beta", 0.25))
    ema_decay = float(section.get("ema_decay", 0.99))
    ema_eps = float(section.get("ema_eps", 1e-5))
    if commitment_beta <= 0:
        raise RuntimeError("Invalid rqvae.commitment_beta in config: must be > 0")
    if ema_decay <= 0 or ema_decay >= 1:
        raise RuntimeError("Invalid rqvae.ema_decay in config: must be > 0 and < 1")
    if ema_eps <= 0:
        raise RuntimeError("Invalid rqvae.ema_eps in config: must be > 0")

    adam_beta1 = float(section.get("adam_beta1", 0.5))
    adam_beta2 = float(section.get("adam_beta2", 0.9))
    if adam_beta1 <= 0 or adam_beta1 >= 1:
        raise RuntimeError("Invalid rqvae.adam_beta1 in config: must be > 0 and < 1")
    if adam_beta2 <= 0 or adam_beta2 >= 1:
        raise RuntimeError("Invalid rqvae.adam_beta2 in config: must be > 0 and < 1")

    checkpoint_every = int(section.get("checkpoint_every", 1))
    if checkpoint_every < 1:
        raise RuntimeError("Invalid rqvae.checkpoint_every in config: must be >= 1")

    encoder_hidden_dim = int(section.get("encoder_hidden_dim", 1024))
    decoder_hidden_dim = int(section.get("decoder_hidden_dim", 1024))
    if encoder_hidden_dim < latent_dim:
        raise RuntimeError("Invalid rqvae.encoder_hidden_dim in config: must be >= latent_dim")
    if decoder_hidden_dim < latent_dim:
        raise RuntimeError("Invalid rqvae.decoder_hidden_dim in config: must be >= latent_dim")

    dry_run = bool(section.get("dry_run", False))
    dry_run_limit = int(section.get("dry_run_limit", 512))
    dry_run_epochs = int(section.get("dry_run_epochs", 1))
    dry_run_batch_size = int(section.get("dry_run_batch_size", 32))
    dry_run_num_workers = int(section.get("dry_run_num_workers", 0))
    dry_run_out_subdir = str(section.get("dry_run_out_subdir", "dry_run")).strip()
    if dry_run_limit < 2:
        raise RuntimeError("Invalid rqvae.dry_run_limit in config: must be >= 2")
    if dry_run_epochs < 1:
        raise RuntimeError("Invalid rqvae.dry_run_epochs in config: must be >= 1")
    if dry_run_batch_size < 2:
        raise RuntimeError("Invalid rqvae.dry_run_batch_size in config: must be >= 2")
    if dry_run_num_workers < 0:
        raise RuntimeError("Invalid rqvae.dry_run_num_workers in config: must be >= 0")
    if not dry_run_out_subdir:
        raise RuntimeError("Invalid rqvae.dry_run_out_subdir in config: must not be empty")

    wandb_enabled = bool(section.get("wandb_enabled", False))
    wandb_mode = str(section.get("wandb_mode", "offline")).strip().lower()
    if wandb_mode not in {"offline", "online", "disabled"}:
        raise RuntimeError(
            "Invalid rqvae.wandb_mode in config: must be one of offline, online, disabled"
        )
    raw_tags = section.get("wandb_tags", [])
    if not isinstance(raw_tags, list):
        raise RuntimeError("Invalid rqvae.wandb_tags in config: must be an array of strings")
    wandb_tags = [str(tag) for tag in raw_tags]

    return RQVAEConfig(
        tokenizer_db=tokenizer_db,
        out_dir=out_dir,
        env_file=env_file,
        rebuild=rebuild,
        limit=limit,
        device=str(section.get("device", "auto")),
        seed=int(section.get("seed", 42)),
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        val_ratio=val_ratio,
        lr=lr,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        warmup_steps=warmup_steps,
        latent_dim=latent_dim,
        rq_levels=rq_levels,
        codebook_size=codebook_size,
        commitment_beta=commitment_beta,
        ema_decay=ema_decay,
        ema_eps=ema_eps,
        restart_unused_codes=bool(section.get("restart_unused_codes", True)),
        amp=bool(section.get("amp", True)),
        checkpoint_every=checkpoint_every,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        dry_run=dry_run,
        dry_run_limit=dry_run_limit,
        dry_run_epochs=dry_run_epochs,
        dry_run_batch_size=dry_run_batch_size,
        dry_run_num_workers=dry_run_num_workers,
        dry_run_out_subdir=dry_run_out_subdir,
        wandb_enabled=wandb_enabled,
        wandb_mode=wandb_mode,
        wandb_project=str(section.get("wandb_project", "")),
        wandb_entity=str(section.get("wandb_entity", "")),
        wandb_api_key=str(section.get("wandb_api_key", "")),
        wandb_run_name=str(section.get("wandb_run_name", "")),
        wandb_group=str(section.get("wandb_group", "")),
        wandb_tags=wandb_tags,
        wandb_project_env=str(section.get("wandb_project_env", "WANDB_PROJECT")),
        wandb_entity_env=str(section.get("wandb_entity_env", "WANDB_ENTITY")),
        wandb_api_key_env=str(section.get("wandb_api_key_env", "WANDB_API_KEY")),
    )


def _load_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")
    with config_path.open("rb") as file:
        parsed = tomllib.load(file)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Invalid config format: {config_path}")
    return parsed


def _require_section(doc: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = doc.get(section_name)
    if not isinstance(section, dict):
        raise RuntimeError(f"Missing required config section: [{section_name}]")
    return section


def _resolve_path(config_path: Path, raw_path: Any) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()
