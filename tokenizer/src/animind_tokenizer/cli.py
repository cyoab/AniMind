from enum import Enum
from pathlib import Path

import typer

from .config import build_embed_config, build_prep_config, build_rqvae_config
from .embed import run_embed
from .prep import run_prep
from .rqvae import run_rqvae

app = typer.Typer(help="AniMind tokenizer pipeline.")


class RunPhase(str, Enum):
    PREP = "prep"
    EMBED = "embed"
    RQVAE = "rqvae"


@app.callback()
def main() -> None:
    """Tokenizer command group."""


@app.command("run")
def run(
    phase: RunPhase = typer.Option(RunPhase.PREP, case_sensitive=False, help="Pipeline phase."),
    config: Path = typer.Option(
        Path("./config/tokenizer.toml"),
        help="TOML config path with [prep], [embedd], and [rqvae] sections.",
    ),
) -> None:
    if phase == RunPhase.PREP:
        prep_config = build_prep_config(config_path=config)
        run_prep(config=prep_config)
        return

    if phase == RunPhase.EMBED:
        embed_config = build_embed_config(config_path=config)
        run_embed(config=embed_config)
        return

    rqvae_config = build_rqvae_config(config_path=config)
    run_rqvae(config=rqvae_config)


if __name__ == "__main__":
    app()
