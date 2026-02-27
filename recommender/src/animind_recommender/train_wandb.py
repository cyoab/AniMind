from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console

from .train_types import WandbConfig


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def resolve_env_value(*, direct_value: str, env_key: str, dotenv_values: dict[str, str]) -> str:
    if direct_value.strip():
        return direct_value.strip()
    if env_key in dotenv_values and dotenv_values[env_key].strip():
        return dotenv_values[env_key].strip()
    return os.environ.get(env_key, "").strip()


def init_wandb(
    *,
    cfg: WandbConfig,
    env_file: Path,
    metadata: dict[str, Any],
    console: Console,
) -> Any | None:
    if not cfg.enabled:
        return None
    if cfg.mode == "disabled":
        return None

    dotenv_values = load_dotenv(env_file)
    project = resolve_env_value(
        direct_value=cfg.project,
        env_key=cfg.project_env,
        dotenv_values=dotenv_values,
    )
    entity = resolve_env_value(
        direct_value=cfg.entity,
        env_key=cfg.entity_env,
        dotenv_values=dotenv_values,
    )
    api_key = resolve_env_value(
        direct_value=cfg.api_key,
        env_key=cfg.api_key_env,
        dotenv_values=dotenv_values,
    )
    if not project:
        raise RuntimeError(
            "W&B is enabled but project is missing. Set train.wandb.project or define "
            f"{cfg.project_env} in {env_file}."
        )
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_MODE"] = cfg.mode

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking enabled but package is not installed. Install with: uv add wandb"
        ) from exc

    run = wandb.init(
        project=project,
        entity=(entity or None),
        mode=cfg.mode,
        name=(cfg.run_name or None),
        group=(cfg.group or None),
        tags=cfg.tags if cfg.tags else None,
        config=metadata,
    )
    console.log(
        "[green]W&B initialized.[/green] "
        f"mode={cfg.mode}, project={project}, entity={entity or 'default'}."
    )
    return run


def wandb_log(*, run: Any | None, payload: dict[str, Any], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


def finish_wandb(*, run: Any | None) -> None:
    if run is None:
        return
    run.finish()
