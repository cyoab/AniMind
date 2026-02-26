from enum import Enum
from pathlib import Path

import typer

from .config import build_prep_config
from .prep import run_prep

app = typer.Typer(help="AniMind recommender pipeline.")


class RunPhase(str, Enum):
    PREP = "prep"


@app.callback()
def main() -> None:
    """Recommender command group."""


@app.command("run")
def run(
    phase: RunPhase = typer.Option(RunPhase.PREP, case_sensitive=False, help="Pipeline phase."),
    config: Path = typer.Option(
        Path("./config/recommender.toml"),
        help="TOML config path with [prep] section.",
    ),
) -> None:
    if phase == RunPhase.PREP:
        prep_config = build_prep_config(config_path=config)
        run_prep(config=prep_config)
        return

    raise RuntimeError(f"Unsupported phase: {phase}")


if __name__ == "__main__":
    app()
