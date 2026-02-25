from enum import Enum
from pathlib import Path

import typer

from .anilist_scraper import KAGGLE_DEFAULT_DATASET, ScrapeConfig, run_scrape

app = typer.Typer(help="AniMind dataset builder.")


class RunPhase(str, Enum):
    ALL = "all"
    CLUBS = "clubs"
    USERS = "users"
    ANIMELIST = "animelist"
    ANIME = "anime"
    EXPORT = "export"


@app.command("scrape")
def scrape(
    out_dir: Path = typer.Option(Path("./output"), help="Output directory."),
    target_possible_users: int = typer.Option(1_000_000, min=1),
    min_club_members: int = typer.Option(30, min=1),
    max_club_pages: int = typer.Option(0, min=0),
    timeout_seconds: float = typer.Option(20.0, min=1.0),
    club_workers: int = typer.Option(1, min=1, help="Phase-2 club workers."),
    club_limit: int = typer.Option(0, min=0, help="Pending clubs to process (0 = all)."),
    club_page_delay_seconds: float = typer.Option(3.0, min=0.0),
    api_delay_seconds: float = typer.Option(4.2, min=0.0),
    jikan_base_url: str = typer.Option("https://api.jikan.moe/v4"),
    export_only: bool = typer.Option(False),
    phase: RunPhase = typer.Option(RunPhase.ALL, case_sensitive=False),
    status_only: bool = typer.Option(False, help="Print resume status and exit."),
    reset_resume: bool = typer.Option(False, help="Clear resume checkpoints."),
    bootstrap_kaggle: bool = typer.Option(False, help="Import baseline Kaggle dataset."),
    kaggle_dataset: str = typer.Option(KAGGLE_DEFAULT_DATASET),
    kaggle_force_download: bool = typer.Option(False),
    bootstrap_path: Path | None = typer.Option(None),
) -> None:
    config = ScrapeConfig(
        output_dir=out_dir,
        target_possible_users=target_possible_users,
        min_club_members=min_club_members,
        max_club_pages=max_club_pages,
        timeout_seconds=timeout_seconds,
        club_workers=club_workers,
        club_limit=club_limit,
        club_page_delay_seconds=club_page_delay_seconds,
        api_delay_seconds=api_delay_seconds,
        jikan_base_url=jikan_base_url,
    )
    run_scrape(
        config=config,
        export_only=export_only,
        phase=phase.value,
        status_only=status_only,
        reset_resume=reset_resume,
        bootstrap_kaggle=bootstrap_kaggle,
        kaggle_dataset=kaggle_dataset,
        kaggle_force_download=kaggle_force_download,
        bootstrap_path=bootstrap_path,
    )


if __name__ == "__main__":
    app()
