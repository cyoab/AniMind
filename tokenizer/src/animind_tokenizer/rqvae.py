from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
from array import array
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from .embed import _cuda_unavailable_reason, _resolve_device


@dataclass(slots=True)
class RQVAEConfig:
    tokenizer_db: Path = Path("./output/tokenizer.sqlite")
    out_dir: Path = Path("./output/rqvae")
    rebuild: bool = True
    limit: int = 0
    device: str = "auto"
    seed: int = 42
    batch_size: int = 256
    epochs: int = 40
    num_workers: int = 4
    val_ratio: float = 0.05
    lr: float = 0.00004
    adam_beta1: float = 0.5
    adam_beta2: float = 0.9
    warmup_steps: int = 1000
    latent_dim: int = 256
    rq_levels: int = 8
    codebook_size: int = 2048
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    restart_unused_codes: bool = True
    amp: bool = True
    checkpoint_every: int = 1
    encoder_hidden_dim: int = 1024
    decoder_hidden_dim: int = 1024
    dry_run: bool = False
    dry_run_limit: int = 512
    dry_run_epochs: int = 1
    dry_run_batch_size: int = 32
    dry_run_num_workers: int = 0
    dry_run_out_subdir: str = "dry_run"
    env_file: Path = Path("./.env")
    wandb_enabled: bool = False
    wandb_mode: str = "offline"
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_api_key: str = ""
    wandb_run_name: str = ""
    wandb_group: str = ""
    wandb_tags: list[str] = field(default_factory=list)
    wandb_project_env: str = "WANDB_PROJECT"
    wandb_entity_env: str = "WANDB_ENTITY"
    wandb_api_key_env: str = "WANDB_API_KEY"
    resume_from: str = ""
    resume_strict: bool = True


class _EncoderMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DecoderMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _SharedCodebookResidualQuantizer(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int,
        latent_dim: int,
        rq_levels: int,
        ema_decay: float,
        ema_eps: float,
        restart_unused_codes: bool,
    ) -> None:
        super().__init__()
        initial_codebook = F.normalize(torch.randn(codebook_size, latent_dim), dim=1)
        self.register_buffer("codebook", initial_codebook)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embed_sum", initial_codebook.clone())
        self.codebook_size = codebook_size
        self.rq_levels = rq_levels
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.restart_unused_codes = restart_unused_codes

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        commitment_loss = z.new_zeros(())
        level_inputs: list[torch.Tensor] = []
        level_indices: list[torch.Tensor] = []
        usage_counts = torch.zeros(self.codebook_size, device=z.device, dtype=torch.float32)

        codebook_sq = (self.codebook**2).sum(dim=1)
        for _ in range(self.rq_levels):
            level_inputs.append(residual.detach())
            residual_sq = (residual**2).sum(dim=1, keepdim=True)
            distances = residual_sq - 2 * (residual @ self.codebook.T) + codebook_sq.unsqueeze(0)
            indices = torch.argmin(distances, dim=1)
            level_indices.append(indices)
            quantized = F.embedding(indices, self.codebook)
            quantized_sum = quantized_sum + quantized
            commitment_loss = commitment_loss + F.mse_loss(residual, quantized.detach())
            residual = residual - quantized
            usage_counts = usage_counts + torch.bincount(indices, minlength=self.codebook_size).to(torch.float32)

        if self.training:
            self._update_codebook_ema(level_inputs=level_inputs, level_indices=level_indices)

        discrete_codes = torch.stack(level_indices, dim=1)
        z_quantized = z + (quantized_sum - z).detach()
        return z_quantized, commitment_loss, discrete_codes, usage_counts

    @torch.no_grad()
    def _update_codebook_ema(self, *, level_inputs: list[torch.Tensor], level_indices: list[torch.Tensor]) -> None:
        targets = torch.cat(level_inputs, dim=0)
        assignments = torch.cat(level_indices, dim=0)
        counts = torch.bincount(assignments, minlength=self.codebook_size).to(self.codebook.dtype)

        embed_sums = torch.zeros_like(self.codebook)
        embed_sums.index_add_(0, assignments, targets)

        decay = self.ema_decay
        self.ema_cluster_size.mul_(decay).add_(counts, alpha=1 - decay)
        self.ema_embed_sum.mul_(decay).add_(embed_sums, alpha=1 - decay)

        total_count = self.ema_cluster_size.sum()
        normalizer = total_count + (self.codebook_size * self.ema_eps)
        smoothed_cluster_size = ((self.ema_cluster_size + self.ema_eps) / normalizer) * total_count
        updated_codebook = self.ema_embed_sum / smoothed_cluster_size.unsqueeze(1)
        self.codebook.copy_(updated_codebook)

        if not self.restart_unused_codes:
            return

        unused_mask = counts <= 0
        if not bool(torch.any(unused_mask)):
            return

        sample_indices = torch.randint(
            low=0,
            high=targets.size(0),
            size=(int(unused_mask.sum().item()),),
            device=targets.device,
        )
        replacement_vectors = targets[sample_indices]
        self.codebook[unused_mask] = replacement_vectors
        self.ema_embed_sum[unused_mask] = replacement_vectors
        self.ema_cluster_size[unused_mask] = 1.0


class _RQVAEModel(nn.Module):
    def __init__(self, *, input_dim: int, config: RQVAEConfig) -> None:
        super().__init__()
        self.encoder = _EncoderMLP(
            input_dim=input_dim,
            hidden_dim=config.encoder_hidden_dim,
            latent_dim=config.latent_dim,
        )
        self.quantizer = _SharedCodebookResidualQuantizer(
            codebook_size=config.codebook_size,
            latent_dim=config.latent_dim,
            rq_levels=config.rq_levels,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
            restart_unused_codes=config.restart_unused_codes,
        )
        self.decoder = _DecoderMLP(
            latent_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden_dim,
            output_dim=input_dim,
        )
        self.commitment_beta = config.commitment_beta

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        z_quantized, commitment_loss, discrete_codes, usage_counts = self.quantizer(z)
        x_recon = self.decoder(z_quantized)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + (self.commitment_beta * commitment_loss)
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "codes": discrete_codes,
            "usage_counts": usage_counts,
        }


def run_rqvae(config: RQVAEConfig) -> None:
    console = Console()
    effective_config = _apply_dry_run_overrides(config=config, console=console)

    torch.manual_seed(effective_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_config.seed)

    requested_device = effective_config.device.strip().lower()
    resolved_device = _resolve_device(device=effective_config.device, torch_module=torch)
    if requested_device == "auto" and resolved_device == "cpu":
        console.log(
            "[yellow]CUDA unavailable; falling back to CPU.[/yellow] "
            f"{_cuda_unavailable_reason(torch_module=torch)}"
        )

    resume_requested = bool(effective_config.resume_from.strip())
    effective_config.out_dir.mkdir(parents=True, exist_ok=True)
    _reset_output_artifacts(config=effective_config, allow_existing=resume_requested)

    stage_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    console.rule("[bold cyan]RQ-VAE Training[/bold cyan]")
    stage_total = 7
    with Progress(*stage_columns, console=console) as stage_progress:
        stage_task = stage_progress.add_task("[cyan]Pipeline stages[/cyan]", total=stage_total)

        stage_progress.update(stage_task, description="[cyan]1/7 Load embeddings[/cyan]")
        dataset = _load_embeddings(config=effective_config)
        stage_progress.advance(stage_task)

        if dataset.size(0) < 2:
            raise RuntimeError("RQ-VAE requires at least 2 embedding rows for train/validation split.")

        stage_progress.update(stage_task, description="[cyan]2/7 Split train/validation[/cyan]")
        train_indices, val_indices = _split_indices(
            total_rows=int(dataset.size(0)),
            val_ratio=effective_config.val_ratio,
            seed=effective_config.seed,
        )
        train_vectors = dataset[train_indices]
        val_vectors = dataset[val_indices]
        input_dim = int(dataset.size(1))
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]3/7 Initialize model/runtime[/cyan]")
        model = _RQVAEModel(input_dim=input_dim, config=effective_config).to(resolved_device)
        gpu_name, gpu_vram = _verify_runtime(resolved_device=resolved_device)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]4/7 Build loaders/optimizer[/cyan]")
        train_loader, val_loader = _build_dataloaders(
            train_vectors=train_vectors,
            val_vectors=val_vectors,
            batch_size=effective_config.batch_size,
            num_workers=effective_config.num_workers,
            pin_memory=resolved_device.startswith("cuda"),
        )
        optimizer = Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=effective_config.lr,
            betas=(effective_config.adam_beta1, effective_config.adam_beta2),
        )
        scheduler = _build_scheduler(optimizer=optimizer, warmup_steps=effective_config.warmup_steps)
        amp_enabled = bool(effective_config.amp and resolved_device.startswith("cuda"))
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
        stage_progress.advance(stage_task)

        metadata = asdict(effective_config)
        metadata["tokenizer_db"] = str(effective_config.tokenizer_db)
        metadata["out_dir"] = str(effective_config.out_dir)
        metadata["env_file"] = str(effective_config.env_file)
        metadata["resolved_device"] = resolved_device
        metadata["dataset_rows"] = int(dataset.size(0))
        metadata["train_rows"] = int(train_vectors.size(0))
        metadata["val_rows"] = int(val_vectors.size(0))
        metadata["input_dim"] = input_dim

        runtime_table = _build_runtime_table(
            config=effective_config,
            resolved_device=resolved_device,
            gpu_name=gpu_name,
            gpu_vram=gpu_vram,
            metadata=metadata,
        )
        console.print(runtime_table)

        stage_progress.update(stage_task, description="[cyan]5/7 Initialize tracking[/cyan]")
        metrics_path = effective_config.out_dir / "rqvae_metrics.jsonl"
        usage_path = effective_config.out_dir / "code_usage.csv"
        _write_json(effective_config.out_dir / "rqvae_config.json", metadata)
        if not (resume_requested and usage_path.exists()):
            _write_usage_header(path=usage_path)
        wandb_run = _init_wandb(config=effective_config, metadata=metadata, console=console)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]6/7 Train model[/cyan]")
        start_epoch = 1
        best_val_loss = math.inf
        if resume_requested:
            start_epoch, best_val_loss = _resume_training_state(
                config=effective_config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_device=resolved_device,
                console=console,
            )
            if start_epoch > effective_config.epochs:
                console.log(
                    "[yellow]Resume checkpoint epoch is already at/above target epochs; "
                    "skipping train loop.[/yellow]"
                )

        epoch_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        last_checkpoint_payload: dict[str, Any] | None = None
        try:
            for epoch in range(start_epoch, effective_config.epochs + 1):
                model.train()
                with Progress(*epoch_columns, console=console) as progress:
                    train_task = progress.add_task(
                        f"[cyan]Epoch {epoch}/{effective_config.epochs} train[/cyan]",
                        total=max(1, len(train_loader)),
                    )
                    train_metrics = _run_epoch(
                        model=model,
                        dataloader=train_loader,
                        device=resolved_device,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                        progress=progress,
                        task_id=train_task,
                    )

                    model.eval()
                    val_task = progress.add_task(
                        f"[magenta]Epoch {epoch}/{effective_config.epochs} val[/magenta]",
                        total=max(1, len(val_loader)),
                    )
                    with torch.no_grad():
                        val_metrics = _run_epoch(
                            model=model,
                            dataloader=val_loader,
                            device=resolved_device,
                            optimizer=None,
                            scheduler=None,
                            scaler=None,
                            amp_enabled=amp_enabled,
                            progress=progress,
                            task_id=val_task,
                        )

                usage_pct, perplexity = _usage_stats(train_metrics["usage_counts"])
                metric_record = {
                    "epoch": epoch,
                    "train_total": train_metrics["loss"],
                    "train_recon": train_metrics["recon"],
                    "train_commit": train_metrics["commit"],
                    "val_total": val_metrics["loss"],
                    "val_recon": val_metrics["recon"],
                    "val_commit": val_metrics["commit"],
                    "code_usage_pct": usage_pct,
                    "code_perplexity": perplexity,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                _append_jsonl(metrics_path, metric_record)
                _append_usage_row(
                    path=usage_path,
                    epoch=epoch,
                    used_codes=int((train_metrics["usage_counts"] > 0).sum().item()),
                    total_codes=effective_config.codebook_size,
                    usage_pct=usage_pct,
                    perplexity=perplexity,
                )
                _wandb_log(wandb_run=wandb_run, payload=metric_record, step=epoch)

                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "config": metadata,
                    "metrics": metric_record,
                }
                last_checkpoint_payload = checkpoint_payload
                if epoch % effective_config.checkpoint_every == 0:
                    _save_checkpoint(
                        path=effective_config.out_dir / "rqvae_last.pt",
                        payload=checkpoint_payload,
                        console=console,
                        label="last",
                    )
                if _should_update_best(
                    epoch=epoch,
                    current_val=float(val_metrics["loss"]),
                    best_val=float(best_val_loss),
                ):
                    best_val_loss = float(val_metrics["loss"])
                    _save_checkpoint(
                        path=effective_config.out_dir / "rqvae_best.pt",
                        payload=checkpoint_payload,
                        console=console,
                        label="best",
                    )
                    best_flag = "yes"
                else:
                    best_flag = "no"

                console.log(
                    "Epoch summary: "
                    f"epoch={epoch}/{effective_config.epochs}, "
                    f"train_total={train_metrics['loss']:.6f}, "
                    f"val_total={val_metrics['loss']:.6f}, "
                    f"usage_pct={usage_pct:.2f}, "
                    f"perplexity={perplexity:.2f}, "
                    f"best_updated={best_flag}."
                )
            if last_checkpoint_payload is not None:
                _save_checkpoint(
                    path=effective_config.out_dir / "rqvae_last.pt",
                    payload=last_checkpoint_payload,
                    console=console,
                    label="last-final",
                )
        except Exception:
            if last_checkpoint_payload is not None:
                _save_checkpoint(
                    path=effective_config.out_dir / "rqvae_last.pt",
                    payload=last_checkpoint_payload,
                    console=console,
                    label="last-recovery",
                )
            raise
        finally:
            _finish_wandb(wandb_run=wandb_run)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]7/7 Finalize/validate artifacts[/cyan]")
        if effective_config.dry_run:
            _validate_dry_run_artifacts(out_dir=effective_config.out_dir, expected_epochs=effective_config.epochs)
            console.log("[green]Dry-run integrity checks passed.[/green]")
        stage_progress.advance(stage_task)

    console.log(
        "RQ-VAE training complete: "
        f"rows={dataset.size(0):,}, input_dim={input_dim}, "
        f"best_checkpoint={effective_config.out_dir / 'rqvae_best.pt'}."
    )


def _apply_dry_run_overrides(*, config: RQVAEConfig, console: Console) -> RQVAEConfig:
    if not config.dry_run:
        return config

    dry_run_limit = max(2, int(config.dry_run_limit))
    limited_rows = dry_run_limit if config.limit <= 0 else min(config.limit, dry_run_limit)
    dry_run_config = replace(
        config,
        limit=limited_rows,
        epochs=max(1, min(config.epochs, config.dry_run_epochs)),
        batch_size=max(2, min(config.batch_size, config.dry_run_batch_size)),
        num_workers=max(0, config.dry_run_num_workers),
        checkpoint_every=1,
        rebuild=True,
        out_dir=config.out_dir / config.dry_run_out_subdir,
    )
    console.log(
        "[yellow]Dry-run mode enabled.[/yellow] "
        f"rows<={dry_run_config.limit}, epochs={dry_run_config.epochs}, "
        f"batch_size={dry_run_config.batch_size}, out_dir={dry_run_config.out_dir}."
    )
    return dry_run_config


def _build_runtime_table(
    *,
    config: RQVAEConfig,
    resolved_device: str,
    gpu_name: str | None,
    gpu_vram: float | None,
    metadata: dict[str, Any],
) -> Table:
    table = Table(title="RQ-VAE Runtime Summary", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Requested Device", config.device)
    table.add_row("Resolved Device", resolved_device)
    table.add_row("Input Dim", str(metadata.get("input_dim", "?")))
    table.add_row("Rows (train/val)", f"{metadata.get('train_rows', '?')}/{metadata.get('val_rows', '?')}")
    table.add_row("Latent / Levels / Codebook", f"{config.latent_dim} / {config.rq_levels} / {config.codebook_size}")
    table.add_row("AMP Enabled", str(bool(config.amp and resolved_device.startswith("cuda"))))
    if gpu_name is not None and gpu_vram is not None:
        table.add_row("GPU", f"{gpu_name} ({gpu_vram:.2f} GiB)")
    table.add_row("W&B", f"enabled={config.wandb_enabled}, mode={config.wandb_mode}")
    return table


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip("'").strip('"')
        values[key.strip()] = cleaned
    return values


def _resolve_env_value(*, direct_value: str, env_key: str, dotenv_values: dict[str, str]) -> str:
    if direct_value.strip():
        return direct_value.strip()
    if env_key in dotenv_values and dotenv_values[env_key].strip():
        return dotenv_values[env_key].strip()
    return os.environ.get(env_key, "").strip()


def _init_wandb(*, config: RQVAEConfig, metadata: dict[str, Any], console: Console) -> Any | None:
    if not config.wandb_enabled:
        return None

    mode = config.wandb_mode.strip().lower()
    if mode not in {"offline", "online", "disabled"}:
        raise RuntimeError("Invalid rqvae.wandb_mode; expected one of: offline, online, disabled.")
    if mode == "disabled":
        return None

    dotenv_values = _load_dotenv(config.env_file)
    project = _resolve_env_value(
        direct_value=config.wandb_project,
        env_key=config.wandb_project_env,
        dotenv_values=dotenv_values,
    )
    entity = _resolve_env_value(
        direct_value=config.wandb_entity,
        env_key=config.wandb_entity_env,
        dotenv_values=dotenv_values,
    )
    api_key = _resolve_env_value(
        direct_value=config.wandb_api_key,
        env_key=config.wandb_api_key_env,
        dotenv_values=dotenv_values,
    )
    if not project:
        raise RuntimeError(
            "W&B is enabled but project is missing. Set rqvae.wandb_project or define "
            f"{config.wandb_project_env} in {config.env_file}."
        )

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = mode

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking enabled but package is not installed. "
            "Install with: uv add wandb"
        ) from exc

    run_name = config.wandb_run_name.strip() or None
    group = config.wandb_group.strip() or None
    tags = [tag.strip() for tag in config.wandb_tags if tag.strip()]
    run = wandb.init(
        project=project,
        entity=(entity or None),
        config=metadata,
        mode=mode,
        name=run_name,
        group=group,
        tags=tags if tags else None,
    )
    console.log(
        "[green]W&B initialized.[/green] "
        f"mode={mode}, project={project}, entity={entity or 'default'}."
    )
    return run


def _wandb_log(*, wandb_run: Any | None, payload: dict[str, Any], step: int) -> None:
    if wandb_run is None:
        return
    wandb_run.log(payload, step=step)


def _finish_wandb(*, wandb_run: Any | None) -> None:
    if wandb_run is None:
        return
    wandb_run.finish()


def _validate_dry_run_artifacts(*, out_dir: Path, expected_epochs: int) -> None:
    required = (
        out_dir / "rqvae_best.pt",
        out_dir / "rqvae_last.pt",
        out_dir / "rqvae_config.json",
        out_dir / "rqvae_metrics.jsonl",
        out_dir / "code_usage.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Dry-run validation failed; missing artifacts: {', '.join(missing)}")

    metric_lines = [
        line for line in (out_dir / "rqvae_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(metric_lines) < expected_epochs:
        raise RuntimeError(
            "Dry-run validation failed; insufficient metric rows in rqvae_metrics.jsonl."
        )

    checkpoint = torch.load(out_dir / "rqvae_best.pt", map_location="cpu")
    required_keys = {"epoch", "model_state_dict", "optimizer_state_dict", "config", "metrics"}
    if not required_keys.issubset(set(checkpoint.keys())):
        raise RuntimeError("Dry-run validation failed; best checkpoint payload is incomplete.")


def _should_update_best(*, epoch: int, current_val: float, best_val: float) -> bool:
    if epoch == 1:
        return True
    if not math.isfinite(current_val):
        return False
    if not math.isfinite(best_val):
        return True
    return current_val < best_val


def _resolve_resume_checkpoint_path(config: RQVAEConfig) -> Path | None:
    raw = config.resume_from.strip()
    if not raw:
        return None
    if raw == "last":
        return config.out_dir / "rqvae_last.pt"
    if raw == "best":
        return config.out_dir / "rqvae_best.pt"
    return Path(raw)


def _resume_training_state(
    *,
    config: RQVAEConfig,
    model: _RQVAEModel,
    optimizer: Adam,
    scheduler: LambdaLR | None,
    map_device: str,
    console: Console,
) -> tuple[int, float]:
    checkpoint_path = _resolve_resume_checkpoint_path(config=config)
    if checkpoint_path is None:
        return 1, math.inf
    if not checkpoint_path.exists():
        if config.resume_strict:
            raise RuntimeError(f"Resume checkpoint not found: {checkpoint_path}")
        console.log(
            f"[yellow]Resume checkpoint not found; continuing fresh run:[/yellow] {checkpoint_path}"
        )
        return 1, math.inf

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Invalid resume checkpoint (missing model_state_dict): {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(map_device)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics")
    best_val = math.inf
    if isinstance(metrics, dict):
        candidate_val = float(metrics.get("val_total", math.inf))
        if math.isfinite(candidate_val):
            best_val = candidate_val

    start_epoch = checkpoint_epoch + 1
    console.log(
        "[green]Resumed from checkpoint.[/green] "
        f"path={checkpoint_path}, checkpoint_epoch={checkpoint_epoch}, next_epoch={start_epoch}."
    )
    return start_epoch, best_val


def load_rqvae_for_eval(*, checkpoint_path: Path, device: str = "cpu") -> tuple[_RQVAEModel, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("config")
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint is missing config metadata for model reconstruction.")
    if "input_dim" not in metadata:
        raise RuntimeError("Checkpoint config is missing input_dim.")

    restored_config = _restore_rqvae_config(metadata=metadata)
    input_dim = int(metadata["input_dim"])
    model = _RQVAEModel(input_dim=input_dim, config=restored_config)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint is missing model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = _resolve_device(device=device, torch_module=torch)
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def _restore_rqvae_config(*, metadata: dict[str, Any]) -> RQVAEConfig:
    valid = {f.name for f in fields(RQVAEConfig)}
    kwargs: dict[str, Any] = {}
    for name in valid:
        if name in metadata:
            kwargs[name] = metadata[name]
    for path_field in ("tokenizer_db", "out_dir", "env_file"):
        if path_field in kwargs:
            kwargs[path_field] = Path(str(kwargs[path_field]))
    if "wandb_tags" in kwargs and kwargs["wandb_tags"] is None:
        kwargs["wandb_tags"] = []
    return RQVAEConfig(**kwargs)


def _run_epoch(
    *,
    model: _RQVAEModel,
    dataloader: DataLoader[tuple[torch.Tensor]],
    device: str,
    optimizer: Adam | None,
    scheduler: LambdaLR | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    progress: Progress,
    task_id: int,
) -> dict[str, torch.Tensor | float]:
    if optimizer is None:
        assert scheduler is None
        assert scaler is None

    sample_count = 0
    total_loss = 0.0
    total_recon = 0.0
    total_commit = 0.0
    usage_counts = torch.zeros(model.quantizer.codebook_size, dtype=torch.float32)

    for batch_idx, (batch_x,) in enumerate(dataloader, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_size = int(batch_x.size(0))
        sample_count += batch_size

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
            if amp_enabled
            else nullcontext()
        ):
            outputs = model(batch_x)
            loss = outputs["loss"]

        loss_value = float(outputs["loss"].detach().item())
        recon_value = float(outputs["recon_loss"].detach().item())
        commit_value = float(outputs["commitment_loss"].detach().item())
        if not (math.isfinite(loss_value) and math.isfinite(recon_value) and math.isfinite(commit_value)):
            raise RuntimeError(
                "Non-finite loss detected during RQ-VAE training. "
                f"batch={batch_idx}, loss={loss_value}, recon={recon_value}, commit={commit_value}. "
                "Check embedding integrity and numeric stability settings."
            )

        if optimizer is not None and scaler is not None:
            scaler.scale(loss).backward()
            _step_optimizer_and_scheduler(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

        total_loss += loss_value * batch_size
        total_recon += recon_value * batch_size
        total_commit += commit_value * batch_size
        usage_counts += outputs["usage_counts"].detach().cpu().to(torch.float32)
        progress.advance(task_id, advance=1)

    if sample_count == 0:
        raise RuntimeError("Encountered an empty dataloader during training.")

    return {
        "loss": total_loss / sample_count,
        "recon": total_recon / sample_count,
        "commit": total_commit / sample_count,
        "usage_counts": usage_counts,
    }


def _build_dataloaders(
    *,
    train_vectors: torch.Tensor,
    val_vectors: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader[tuple[torch.Tensor]], DataLoader[tuple[torch.Tensor]]]:
    train_loader = DataLoader(
        TensorDataset(train_vectors),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        TensorDataset(val_vectors),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def _step_optimizer_and_scheduler(
    *,
    optimizer: Adam,
    scheduler: LambdaLR | None,
    scaler: torch.amp.GradScaler,
) -> bool:
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())

    optimizer_stepped = scale_after >= scale_before
    if scheduler is not None and optimizer_stepped:
        scheduler.step()
    return optimizer_stepped


def _build_scheduler(*, optimizer: Adam, warmup_steps: int) -> LambdaLR | None:
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step >= warmup_steps:
            return 1.0
        return float(step + 1) / float(warmup_steps)

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def _load_embeddings(config: RQVAEConfig) -> torch.Tensor:
    if not config.tokenizer_db.exists():
        raise RuntimeError(f"Tokenizer DB not found: {config.tokenizer_db}")

    query = "SELECT anime_id, embedding, embedding_dim FROM anime_embeddings ORDER BY anime_id"
    if config.limit > 0:
        query += f" LIMIT {int(config.limit)}"

    vectors: list[torch.Tensor] = []
    expected_dim: int | None = None
    non_finite_ids: list[int] = []
    with sqlite3.connect(config.tokenizer_db) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")
        if not _table_exists(conn, "anime_embeddings"):
            raise RuntimeError("Missing anime_embeddings table. Run --phase embed before --phase rqvae.")

        cursor = conn.execute(query)
        for anime_id, blob, embedding_dim in cursor:
            if expected_dim is None:
                expected_dim = int(embedding_dim)
            elif int(embedding_dim) != expected_dim:
                raise RuntimeError(
                    "anime_embeddings has inconsistent embedding_dim metadata across rows."
                )
            vector = _blob_to_tensor(blob=blob)
            if int(vector.numel()) != expected_dim:
                raise RuntimeError(
                    "Embedding blob length does not match embedding_dim metadata."
                )
            if not bool(torch.isfinite(vector).all()):
                non_finite_ids.append(int(anime_id))
                if len(non_finite_ids) >= 20:
                    break
                continue
            vectors.append(vector)

    if non_finite_ids:
        preview = ", ".join(str(item) for item in non_finite_ids[:10])
        raise RuntimeError(
            "anime_embeddings contains non-finite values (NaN/Inf), so RQ-VAE training cannot proceed. "
            f"example_anime_ids=[{preview}]. "
            "Re-run --phase embed after enabling stable precision and verifying finite embeddings."
        )
    if not vectors:
        raise RuntimeError("No embeddings found in anime_embeddings; nothing to train.")

    return torch.stack(vectors, dim=0)


def _blob_to_tensor(blob: bytes) -> torch.Tensor:
    unpacked = array("f")
    unpacked.frombytes(blob)
    return torch.tensor(unpacked, dtype=torch.float32)


def _split_indices(*, total_rows: int, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if total_rows < 2:
        raise RuntimeError("Need at least two rows to create train/validation split.")
    raw_val_size = int(round(total_rows * val_ratio))
    val_size = min(max(1, raw_val_size), total_rows - 1)
    generator = torch.Generator().manual_seed(seed)
    permuted = torch.randperm(total_rows, generator=generator)
    val_indices = permuted[:val_size]
    train_indices = permuted[val_size:]
    return train_indices, val_indices


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _usage_stats(usage_counts: torch.Tensor) -> tuple[float, float]:
    total_codes = int(usage_counts.numel())
    used = int((usage_counts > 0).sum().item())
    usage_pct = (float(used) / float(total_codes)) * 100 if total_codes > 0 else 0.0
    total_count = float(usage_counts.sum().item())
    if total_count <= 0:
        return usage_pct, 0.0

    probs = usage_counts / total_count
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs)).item()
    perplexity = float(math.exp(entropy))
    return usage_pct, perplexity


def _verify_runtime(resolved_device: str) -> tuple[str | None, float | None]:
    if not resolved_device.startswith("cuda"):
        return None, None
    try:
        target_device = torch.device(resolved_device)
        device_index = (
            int(target_device.index)
            if target_device.index is not None
            else int(torch.cuda.current_device())
        )
        props = torch.cuda.get_device_properties(device_index)
        probe = torch.randn((256, 256), device=resolved_device, dtype=torch.float16)
        _ = (probe @ probe.T).sum().item()
        torch.cuda.synchronize()
        return props.name, props.total_memory / (1024**3)
    except Exception as exc:
        raise RuntimeError(
            "CUDA runtime probe failed while initializing RQ-VAE. "
            "Check pod GPU passthrough, driver, and CUDA runtime compatibility."
        ) from exc


def _save_checkpoint(*, path: Path, payload: dict[str, Any], console: Console, label: str) -> None:
    with console.status(f"[cyan]Saving {label} checkpoint...[/cyan]", spinner="dots"):
        torch.save(payload, path)
    console.log(f"[green]Checkpoint saved:[/green] {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True))
        file.write("\n")


def _write_usage_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "used_codes", "total_codes", "usage_pct", "perplexity"])


def _append_usage_row(
    *,
    path: Path,
    epoch: int,
    used_codes: int,
    total_codes: int,
    usage_pct: float,
    perplexity: float,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, used_codes, total_codes, f"{usage_pct:.6f}", f"{perplexity:.6f}"])


def _reset_output_artifacts(config: RQVAEConfig, *, allow_existing: bool = False) -> None:
    managed_files = (
        config.out_dir / "rqvae_best.pt",
        config.out_dir / "rqvae_last.pt",
        config.out_dir / "rqvae_config.json",
        config.out_dir / "rqvae_metrics.jsonl",
        config.out_dir / "code_usage.csv",
    )
    if not config.rebuild:
        existing = [path for path in managed_files if path.exists()]
        if existing:
            if allow_existing:
                return
            joined = ", ".join(str(path) for path in existing)
            raise RuntimeError(
                "RQ-VAE output artifacts already exist and rebuild=false. "
                f"Set rebuild=true or clean these files: {joined}"
            )
        return

    if allow_existing:
        return

    for path in managed_files:
        if path.exists():
            path.unlink()
