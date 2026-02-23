from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import typer

from .config import Settings
from .export_parquet import export_run_to_parquet
from .pipeline import run_build
from .storage_sqlite import DatasetStorage


app = typer.Typer(help="Build anime preference datasets from Jikan v4", no_args_is_help=True)
console = Console()


def _open_storage(out_dir: Path) -> DatasetStorage:
    db_path = out_dir / "state.sqlite3"
    if not db_path.exists():
        raise typer.BadParameter(f"SQLite state file not found at {db_path}")
    storage = DatasetStorage(db_path)
    storage.initialize()
    return storage


def _print_stats_table(storage: DatasetStorage, run_id: str) -> None:
    stats = storage.get_stats(run_id)

    table = Table(title=f"Run Stats: {run_id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Status", stats.status)
    table.add_row("Target users", str(stats.target_users))
    table.add_row("Users discovered", str(stats.users_discovered))
    table.add_row("Users success", str(stats.users_success))
    table.add_row("Users private", str(stats.users_private))
    table.add_row("Users not found", str(stats.users_not_found))
    table.add_row("Users errors", str(stats.users_errors))
    table.add_row("User anime rows", str(stats.user_anime_rows))
    table.add_row("Completed ratings rows", str(stats.completed_ratings_rows))
    table.add_row("Queued anime", str(stats.queued_anime))
    table.add_row("Processed anime", str(stats.processed_anime))
    table.add_row("Anime rows", str(stats.anime_rows))
    table.add_row("Anime statistics rows", str(stats.anime_stats_rows))
    table.add_row("Anime staff rows", str(stats.anime_staff_rows))
    table.add_row("Anime reviews rows", str(stats.anime_reviews_rows))

    console.print(table)
    if stats.user_anime_rows == 0:
        console.print(
            "[bold yellow]Warning:[/bold yellow] no user anime rows were ingested for this run. "
            "Dataset export is incomplete."
        )


@app.command()
def build(
    target_users: int = typer.Option(10000, min=1),
    out_dir: Path = typer.Option(Path("./output"), file_okay=False, dir_okay=True, writable=True),
    include_nsfw: bool = typer.Option(False, "--include-nsfw/--exclude-nsfw"),
    review_limit: int = typer.Option(50, min=1, max=200),
    run_id: str | None = typer.Option(None),
) -> None:
    """Discover users, ingest user preferences, enrich anime metadata, and export parquet files."""

    settings = Settings()
    out_dir = out_dir.resolve()
    target_discovery = max(target_users, int(target_users * (1.0 + settings.user_buffer_ratio)))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    discovery_task = progress.add_task("Discover users", total=max(1, target_discovery))
    users_task = progress.add_task("Ingest user lists", total=max(1, target_users))
    anime_task = progress.add_task("Enrich anime", total=1, visible=False)

    def discovery_callback(found: int, goal: int, page: int, source: str) -> None:
        progress.update(
            discovery_task,
            description=f"Discover users [{source} p={page}]",
            total=max(1, goal),
            completed=min(found, goal),
        )

    def user_callback(user_progress, total: int) -> None:
        progress.update(
            users_task,
            description=(
                "Ingest user lists "
                f"[ok={user_progress.success} private={user_progress.private} "
                f"not_found={user_progress.not_found} err={user_progress.errors}]"
            ),
            total=max(1, total),
            completed=min(user_progress.processed, max(1, total)),
        )

    def anime_callback(anime_progress, total: int) -> None:
        progress.update(
            anime_task,
            visible=True,
            description=(
                "Enrich anime "
                f"[ok={anime_progress.success} skipped_nsfw={anime_progress.skipped_nsfw} "
                f"err={anime_progress.errors} reviews={anime_progress.reviews_ingested}]"
            ),
            total=max(1, total),
            completed=min(anime_progress.processed, max(1, total)),
        )

    with progress:
        result = asyncio.run(
            run_build(
                out_dir=out_dir,
                target_users=target_users,
                include_nsfw=include_nsfw,
                review_limit=review_limit,
                run_id=run_id,
                settings=settings,
                discovery_callback=discovery_callback,
                user_callback=user_callback,
                anime_callback=anime_callback,
            )
        )

    summary = Table(title="Build Summary")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Run id", result.run_id)
    summary.add_row("Output", str(out_dir))
    summary.add_row("Requests", str(result.telemetry.total_requests))
    summary.add_row("Retries", str(result.telemetry.retries))
    summary.add_row("429 responses", str(result.telemetry.rate_limited))
    summary.add_row("Mean latency (ms)", f"{result.telemetry.mean_latency_ms:.1f}")
    summary.add_row("User rows ingested", str(result.user_progress.success))
    summary.add_row("Anime rows ingested", str(result.anime_progress.success))
    summary.add_row("Reviews ingested", str(result.anime_progress.reviews_ingested))
    console.print(summary)

    storage = _open_storage(out_dir)
    _print_stats_table(storage, result.run_id)
    storage.close()


@app.command()
def export(
    run_id: str = typer.Option(...),
    out_dir: Path = typer.Option(Path("./output"), file_okay=False, dir_okay=True, writable=True),
) -> None:
    """Regenerate parquet datasets and run manifest from SQLite state."""

    out_dir = out_dir.resolve()
    db_path = out_dir / "state.sqlite3"
    if not db_path.exists():
        raise typer.BadParameter(f"SQLite state file not found at {db_path}")

    manifest = export_run_to_parquet(db_path=db_path, run_id=run_id, out_dir=out_dir)

    table = Table(title="Export Summary")
    table.add_column("Dataset")
    table.add_column("Rows", justify="right")
    for dataset, count in manifest["counts"].items():
        table.add_row(dataset, str(count))
    console.print(table)
    console.print(f"Manifest: {out_dir / 'run_manifest.json'}")


@app.command()
def stats(
    run_id: str | None = typer.Option(None),
    out_dir: Path = typer.Option(Path("./output"), file_okay=False, dir_okay=True, writable=True),
) -> None:
    """Print run health, coverage, and failure counts."""

    out_dir = out_dir.resolve()
    storage = _open_storage(out_dir)
    resolved_run_id = run_id or storage.latest_run_id()
    if resolved_run_id is None:
        storage.close()
        raise typer.BadParameter("No runs found in state database")

    _print_stats_table(storage, resolved_run_id)
    storage.close()


if __name__ == "__main__":
    app()
