from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd


def _read_query(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=params)


def export_run_to_parquet(*, db_path: Path, run_id: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    datasets = {
        "users": (
            """
            SELECT run_id, username, mal_id, profile_url, images_json, joined, last_online,
                   is_public, discovered_at, crawl_state, crawl_error
            FROM users
            WHERE run_id=?
            """,
            "users.parquet",
        ),
        "user_anime_list": (
            """
            SELECT run_id, username, anime_mal_id, status, score, episodes_watched, updated_at
            FROM user_anime_list
            WHERE run_id=?
            """,
            "user_anime_list.parquet",
        ),
        "completed_ratings": (
            """
            SELECT run_id, username, anime_mal_id, score, updated_at
            FROM user_anime_list
            WHERE run_id=? AND status='completed' AND score > 0
            """,
            "completed_ratings.parquet",
        ),
        "anime": (
            """
            SELECT run_id, mal_id, title, titles_json, type, source, episodes, status, duration,
                   rating, synopsis, genres_json, themes_json, demographics_json,
                   studios_json, producers_json, nsfw, fetched_at
            FROM anime
            WHERE run_id=?
            """,
            "anime.parquet",
        ),
        "anime_statistics": (
            """
            SELECT run_id, mal_id, watching, completed, on_hold, dropped,
                   plan_to_watch, total, scores_distribution_json, fetched_at
            FROM anime_statistics
            WHERE run_id=?
            """,
            "anime_statistics.parquet",
        ),
        "anime_staff": (
            """
            SELECT run_id, mal_id, person_mal_id, name, positions_json
            FROM anime_staff
            WHERE run_id=?
            """,
            "anime_staff.parquet",
        ),
        "anime_reviews": (
            """
            SELECT run_id, review_key, review_mal_id, anime_mal_id, username, score,
                   tags_json, is_spoiler, is_preliminary, review, reactions_json, created_at
            FROM anime_reviews
            WHERE run_id=?
            """,
            "anime_reviews.parquet",
        ),
    }

    counts: dict[str, int] = {}
    for name, (query, filename) in datasets.items():
        df = _read_query(conn, query, (run_id,))
        df.to_parquet(out_dir / filename, index=False)
        counts[name] = int(len(df))

    quality = {
        "null_usernames": int(
            _read_query(
                conn,
                "SELECT COUNT(*) AS c FROM users WHERE run_id=? AND (username IS NULL OR username='')",
                (run_id,),
            )["c"].iloc[0]
        ),
        "duplicate_user_anime_rows": int(
            _read_query(
                conn,
                """
                SELECT COUNT(*) AS c
                FROM (
                    SELECT username, anime_mal_id, COUNT(*) AS n
                    FROM user_anime_list
                    WHERE run_id=?
                    GROUP BY username, anime_mal_id
                    HAVING n > 1
                )
                """,
                (run_id,),
            )["c"].iloc[0]
        ),
        "completed_without_anime": int(
            _read_query(
                conn,
                """
                SELECT COUNT(*) AS c
                FROM user_anime_list u
                LEFT JOIN anime a ON a.run_id = u.run_id AND a.mal_id = u.anime_mal_id
                WHERE u.run_id=? AND u.status='completed' AND u.score > 0 AND a.mal_id IS NULL
                """,
                (run_id,),
            )["c"].iloc[0]
        ),
    }

    run_df = _read_query(
        conn,
        "SELECT run_id, target_users, include_nsfw, review_limit, started_at, finished_at, status FROM runs WHERE run_id=?",
        (run_id,),
    )
    if run_df.empty:
        raise ValueError(f"Run id not found: {run_id}")
    run_raw = run_df.iloc[0].to_dict()
    run: dict[str, Any] = {}
    for key, value in run_raw.items():
        if pd.isna(value):
            run[key] = None
            continue
        if hasattr(value, "item"):
            try:
                run[key] = value.item()
                continue
            except Exception:
                pass
        run[key] = value

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "run": run,
        "counts": counts,
        "quality_checks": quality,
    }

    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    conn.close()
    return manifest
