from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from rich.console import Console

TEXT_NULL_LITERALS = {"", "nan", "none", "null"}
TEXT_UNKNOWN_LITERALS = {"unknown", "n/a"}
PREP_PHASE = Literal["prep"]
ANIME_COLUMNS = (
    "anime_id",
    "name",
    "english_name",
    "japanese_name",
    "title_synonyms",
    "score",
    "genres",
    "synopsis",
    "type",
    "episodes",
    "premiered",
    "season",
    "year",
    "studios",
    "producers",
    "licensors",
    "source",
    "anime_status",
    "rating",
    "duration",
    "aired_from",
    "aired_to",
    "members",
    "favorites",
    "scored_by",
    "rank",
    "popularity",
    "imported_at",
)


@dataclass(slots=True)
class PrepConfig:
    source_db: Path = Path("../output/anilist.sqlite")
    out_dir: Path = Path("./output")
    rebuild: bool = True
    export_parquet: bool = True
    limit: int = 0

    @property
    def output_db(self) -> Path:
        return self.out_dir / "tokenizer.sqlite"


def run_pipeline(config: PrepConfig, phase: PREP_PHASE = "prep") -> None:
    if phase != "prep":
        raise ValueError(f"Unsupported phase: {phase}")
    run_prep(config=config)


def run_prep(config: PrepConfig) -> None:
    console = Console()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    prepared_at = _now()

    source_uri = f"file:{config.source_db}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_conn, sqlite3.connect(
        config.output_db
    ) as target_conn:
        source_conn.row_factory = sqlite3.Row
        source_conn.execute("PRAGMA busy_timeout = 8000;")
        target_conn.execute("PRAGMA busy_timeout = 8000;")

        if not _table_exists(source_conn, "anime"):
            raise RuntimeError(f"Source database missing required table: anime ({config.source_db})")

        if not config.rebuild and _table_exists(target_conn, "anime_prep"):
            existing_rows = _table_row_count(target_conn, "anime_prep")
            if existing_rows > 0:
                console.log(f"[yellow]Reuse enabled:[/yellow] keeping existing anime_prep ({existing_rows:,} rows).")
                return

        source_df = _load_anime_source(source_conn=source_conn, limit=config.limit)
        cleaned_rows = [_clean_anime_row(row=row, prepared_at=prepared_at) for row in source_df.to_dict("records")]

        _create_target_table(target_conn=target_conn, rebuild=config.rebuild)
        _insert_rows(target_conn=target_conn, rows=cleaned_rows)

        if config.export_parquet:
            export_df = pd.read_sql_query("SELECT * FROM anime_prep", target_conn)
            parquet_path = config.out_dir / "anime_prep.parquet"
            export_df.to_parquet(parquet_path, index=False)
            console.log(f"[green]Wrote[/green] {parquet_path}")

        rows_written = len(cleaned_rows)
        missing_core_count = sum(row["flag_missing_core"] for row in cleaned_rows)
        missing_synopsis_count = sum(row["flag_missing_synopsis"] for row in cleaned_rows)
        console.log(
            "Prep complete: "
            f"source_rows={len(source_df):,}, "
            f"rows_written={rows_written:,}, "
            f"missing_core={missing_core_count:,}, "
            f"missing_synopsis={missing_synopsis_count:,}."
        )


def _load_anime_source(source_conn: sqlite3.Connection, limit: int) -> pd.DataFrame:
    select_columns = ", ".join(ANIME_COLUMNS)
    query = f"SELECT {select_columns} FROM anime ORDER BY anime_id"
    if limit > 0:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, source_conn)
    missing_columns = [col for col in ANIME_COLUMNS if col not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise RuntimeError(f"Source anime table missing required columns: {joined}")
    return df


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def _table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _create_target_table(target_conn: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        target_conn.execute("DROP TABLE IF EXISTS anime_prep")
    target_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anime_prep (
            anime_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            english_name TEXT,
            japanese_name TEXT,
            title_synonyms TEXT,
            score REAL,
            genres TEXT,
            synopsis TEXT,
            type TEXT,
            episodes INTEGER,
            premiered TEXT,
            season TEXT,
            year INTEGER,
            studios TEXT,
            producers TEXT,
            licensors TEXT,
            source TEXT,
            anime_status TEXT,
            rating TEXT,
            duration TEXT,
            aired_from TEXT,
            aired_to TEXT,
            members INTEGER,
            favorites INTEGER,
            scored_by INTEGER,
            rank INTEGER,
            popularity INTEGER,
            flag_missing_core INTEGER NOT NULL,
            flag_missing_synopsis INTEGER NOT NULL,
            anime_text TEXT NOT NULL,
            prepared_at TEXT NOT NULL,
            source_imported_at TEXT
        )
        """
    )
    target_conn.commit()


def _insert_rows(target_conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    target_conn.executemany(
        """
        INSERT OR REPLACE INTO anime_prep (
            anime_id, name, english_name, japanese_name, title_synonyms, score, genres, synopsis, type, episodes,
            premiered, season, year, studios, producers, licensors, source, anime_status, rating, duration, aired_from, aired_to,
            members, favorites, scored_by, rank, popularity,
            flag_missing_core, flag_missing_synopsis, anime_text, prepared_at, source_imported_at
        ) VALUES (
            :anime_id, :name, :english_name, :japanese_name, :title_synonyms, :score, :genres, :synopsis, :type, :episodes,
            :premiered, :season, :year, :studios, :producers, :licensors, :source, :anime_status, :rating, :duration, :aired_from, :aired_to,
            :members, :favorites, :scored_by, :rank, :popularity,
            :flag_missing_core, :flag_missing_synopsis, :anime_text, :prepared_at, :source_imported_at
        )
        """,
        rows,
    )
    target_conn.commit()


def _clean_anime_row(row: dict[str, Any], prepared_at: str) -> dict[str, Any]:
    anime_id = _coerce_nonneg_int(row.get("anime_id"))
    if anime_id is None:
        raise RuntimeError("Encountered anime row with invalid anime_id.")

    cleaned_name = _clean_text(row.get("name"))
    cleaned_english_name = _clean_text(row.get("english_name"))
    cleaned_japanese_name = _clean_text(row.get("japanese_name"))
    cleaned_title_synonyms = _clean_text(row.get("title_synonyms"))
    cleaned_score = _clean_score(row.get("score"))
    cleaned_genres = _clean_text(row.get("genres"))
    cleaned_synopsis = _clean_text(row.get("synopsis"))
    cleaned_type = _clean_text(row.get("type"), treat_unknown_as_null=True)
    cleaned_episodes = _coerce_nonneg_int(row.get("episodes"))
    cleaned_premiered = _clean_text(row.get("premiered"))
    cleaned_season = _clean_text(row.get("season"))
    cleaned_year = _coerce_nonneg_int(row.get("year"))
    cleaned_studios = _clean_text(row.get("studios"))
    cleaned_producers = _clean_text(row.get("producers"))
    cleaned_licensors = _clean_text(row.get("licensors"))
    cleaned_source = _clean_text(row.get("source"))
    cleaned_anime_status = _clean_text(row.get("anime_status"), treat_unknown_as_null=True)
    cleaned_rating = _clean_text(row.get("rating"), treat_unknown_as_null=True)
    cleaned_duration = _clean_text(row.get("duration"))
    cleaned_aired_from = _clean_text(row.get("aired_from"))
    cleaned_aired_to = _clean_text(row.get("aired_to"))
    cleaned_members = _coerce_nonneg_int(row.get("members"))
    cleaned_favorites = _coerce_nonneg_int(row.get("favorites"))
    cleaned_scored_by = _coerce_nonneg_int(row.get("scored_by"))
    cleaned_rank = _coerce_nonneg_int(row.get("rank"))
    cleaned_popularity = _coerce_nonneg_int(row.get("popularity"))
    source_imported_at = _clean_text(row.get("imported_at"))

    flag_missing_core = int(
        any(
            value is None
            for value in (cleaned_name, cleaned_type, cleaned_episodes, cleaned_rating, cleaned_members)
        )
    )
    flag_missing_synopsis = int(cleaned_synopsis is None)

    storage_name = cleaned_name or "Unknown"
    anime_text = _build_anime_text(
        anime_id=anime_id,
        name=storage_name,
        english_name=cleaned_english_name,
        japanese_name=cleaned_japanese_name,
        title_synonyms=cleaned_title_synonyms,
        score=cleaned_score,
        genres=cleaned_genres,
        synopsis=cleaned_synopsis,
        anime_type=cleaned_type,
        episodes=cleaned_episodes,
        premiered=cleaned_premiered,
        season=cleaned_season,
        year=cleaned_year,
        studios=cleaned_studios,
        producers=cleaned_producers,
        licensors=cleaned_licensors,
        source=cleaned_source,
        anime_status=cleaned_anime_status,
        rating=cleaned_rating,
        duration=cleaned_duration,
        aired_from=cleaned_aired_from,
        aired_to=cleaned_aired_to,
        members=cleaned_members,
        favorites=cleaned_favorites,
        scored_by=cleaned_scored_by,
        rank=cleaned_rank,
        popularity=cleaned_popularity,
    )

    return {
        "anime_id": anime_id,
        "name": storage_name,
        "english_name": cleaned_english_name,
        "japanese_name": cleaned_japanese_name,
        "title_synonyms": cleaned_title_synonyms,
        "score": cleaned_score,
        "genres": cleaned_genres,
        "synopsis": cleaned_synopsis,
        "type": cleaned_type,
        "episodes": cleaned_episodes,
        "premiered": cleaned_premiered,
        "season": cleaned_season,
        "year": cleaned_year,
        "studios": cleaned_studios,
        "producers": cleaned_producers,
        "licensors": cleaned_licensors,
        "source": cleaned_source,
        "anime_status": cleaned_anime_status,
        "rating": cleaned_rating,
        "duration": cleaned_duration,
        "aired_from": cleaned_aired_from,
        "aired_to": cleaned_aired_to,
        "members": cleaned_members,
        "favorites": cleaned_favorites,
        "scored_by": cleaned_scored_by,
        "rank": cleaned_rank,
        "popularity": cleaned_popularity,
        "flag_missing_core": flag_missing_core,
        "flag_missing_synopsis": flag_missing_synopsis,
        "anime_text": anime_text,
        "prepared_at": prepared_at,
        "source_imported_at": source_imported_at,
    }


def _clean_text(value: Any, *, treat_unknown_as_null: bool = False) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in TEXT_NULL_LITERALS:
        return None
    if treat_unknown_as_null and text.lower() in TEXT_UNKNOWN_LITERALS:
        return None
    return text.replace(";", ",")


def _clean_score(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = float(value)
        if parsed < 0 or parsed > 10:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _coerce_nonneg_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = int(float(value))
        if parsed < 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, float):
        formatted = f"{value:.6f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return str(value)


def _build_anime_text(
    *,
    anime_id: int,
    name: str,
    english_name: str | None,
    japanese_name: str | None,
    title_synonyms: str | None,
    score: float | None,
    genres: str | None,
    synopsis: str | None,
    anime_type: str | None,
    episodes: int | None,
    premiered: str | None,
    season: str | None,
    year: int | None,
    studios: str | None,
    producers: str | None,
    licensors: str | None,
    source: str | None,
    anime_status: str | None,
    rating: str | None,
    duration: str | None,
    aired_from: str | None,
    aired_to: str | None,
    members: int | None,
    favorites: int | None,
    scored_by: int | None,
    rank: int | None,
    popularity: int | None,
) -> str:
    parts = [
        ("Anime ID", anime_id),
        ("Name", name),
        ("English Name", english_name),
        ("Japanese Name", japanese_name),
        ("Title Synonyms", title_synonyms),
        ("Score", score),
        ("Genres", genres),
        ("Synopsis", synopsis),
        ("Type", anime_type),
        ("Episodes", episodes),
        ("Premiered", premiered),
        ("Season", season),
        ("Year", year),
        ("Studios", studios),
        ("Producers", producers),
        ("Licensors", licensors),
        ("Source", source),
        ("Status", anime_status),
        ("Rating", rating),
        ("Duration", duration),
        ("Aired From", aired_from),
        ("Aired To", aired_to),
        ("Members", members),
        ("Favorites", favorites),
        ("Scored By", scored_by),
        ("Rank", rank),
        ("Popularity", popularity),
    ]
    return "; ".join(f"{label}: {_format_value(value)}" for label, value in parts) + "."


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
