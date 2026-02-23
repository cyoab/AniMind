from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from .models import (
    AnimeCore,
    AnimeReviewEntry,
    AnimeStaffEntry,
    AnimeStats,
    UserAnimeEntry,
    UserProfile,
)


RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"


@dataclass(slots=True)
class RunStats:
    run_id: str
    target_users: int
    status: str
    started_at: str
    finished_at: str | None
    users_discovered: int
    users_success: int
    users_private: int
    users_not_found: int
    users_errors: int
    user_anime_rows: int
    completed_ratings_rows: int
    queued_anime: int
    processed_anime: int
    anime_rows: int
    anime_stats_rows: int
    anime_staff_rows: int
    anime_reviews_rows: int


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DatasetStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                target_users INTEGER NOT NULL,
                include_nsfw INTEGER NOT NULL,
                review_limit INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, key),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS users (
                run_id TEXT NOT NULL,
                username TEXT NOT NULL,
                mal_id INTEGER,
                profile_url TEXT,
                images_json TEXT,
                joined TEXT,
                last_online TEXT,
                is_public INTEGER,
                discovered_at TEXT NOT NULL,
                crawl_state TEXT NOT NULL DEFAULT 'pending',
                crawl_error TEXT,
                PRIMARY KEY (run_id, username),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_anime_list (
                run_id TEXT NOT NULL,
                username TEXT NOT NULL,
                anime_mal_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                episodes_watched INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY (run_id, username, anime_mal_id),
                FOREIGN KEY (run_id, username) REFERENCES users(run_id, username) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_reviews (
                run_id TEXT NOT NULL,
                username TEXT NOT NULL,
                review_mal_id INTEGER,
                anime_mal_id INTEGER,
                score INTEGER,
                review TEXT,
                created_at TEXT,
                PRIMARY KEY (run_id, username, review_mal_id)
            );

            CREATE TABLE IF NOT EXISTS anime_queue (
                run_id TEXT NOT NULL,
                anime_mal_id INTEGER NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                last_state TEXT,
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, anime_mal_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS anime (
                run_id TEXT NOT NULL,
                mal_id INTEGER NOT NULL,
                title TEXT,
                titles_json TEXT,
                type TEXT,
                source TEXT,
                episodes INTEGER,
                status TEXT,
                duration TEXT,
                rating TEXT,
                synopsis TEXT,
                genres_json TEXT,
                themes_json TEXT,
                demographics_json TEXT,
                studios_json TEXT,
                producers_json TEXT,
                nsfw INTEGER NOT NULL DEFAULT 0,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (run_id, mal_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS anime_statistics (
                run_id TEXT NOT NULL,
                mal_id INTEGER NOT NULL,
                watching INTEGER NOT NULL,
                completed INTEGER NOT NULL,
                on_hold INTEGER NOT NULL,
                dropped INTEGER NOT NULL,
                plan_to_watch INTEGER NOT NULL,
                total INTEGER NOT NULL,
                scores_distribution_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (run_id, mal_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS anime_staff (
                run_id TEXT NOT NULL,
                mal_id INTEGER NOT NULL,
                person_mal_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                positions_json TEXT NOT NULL,
                PRIMARY KEY (run_id, mal_id, person_mal_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS anime_reviews (
                run_id TEXT NOT NULL,
                review_key TEXT NOT NULL,
                review_mal_id INTEGER,
                anime_mal_id INTEGER NOT NULL,
                username TEXT,
                score INTEGER,
                tags_json TEXT NOT NULL,
                is_spoiler INTEGER NOT NULL,
                is_preliminary INTEGER NOT NULL,
                review TEXT,
                reactions_json TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (run_id, review_key),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_users_run_state ON users(run_id, crawl_state);
            CREATE INDEX IF NOT EXISTS idx_user_anime_run_anime ON user_anime_list(run_id, anime_mal_id);
            CREATE INDEX IF NOT EXISTS idx_anime_queue_run_processed ON anime_queue(run_id, processed, frequency DESC);
            CREATE INDEX IF NOT EXISTS idx_reviews_run_anime ON anime_reviews(run_id, anime_mal_id);
            """
        )
        self.conn.commit()

    def create_run(self, run_id: str, target_users: int, include_nsfw: bool, review_limit: int) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO runs(run_id, target_users, include_nsfw, review_limit, started_at, status)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (run_id, target_users, int(include_nsfw), review_limit, utc_now_iso(), RUNNING),
        )
        self.conn.commit()

    def set_run_status(self, run_id: str, status: str) -> None:
        finished_at = utc_now_iso() if status in (COMPLETED, FAILED) else None
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=COALESCE(?, finished_at) WHERE run_id=?",
            (status, finished_at, run_id),
        )
        self.conn.commit()

    def run_exists(self, run_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return row is not None

    def set_checkpoint(self, run_id: str, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO checkpoints(run_id, key, value, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(run_id, key)
            DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (run_id, key, value, utc_now_iso()),
        )
        self.conn.commit()

    def get_checkpoint(self, run_id: str, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM checkpoints WHERE run_id=? AND key=?",
            (run_id, key),
        ).fetchone()
        return row["value"] if row else default

    def upsert_discovered_users(self, run_id: str, users_payload: list[dict[str, Any]]) -> int:
        if not users_payload:
            return 0
        before = self.count_users(run_id)
        now = utc_now_iso()
        rows = [
            (
                run_id,
                str(item.get("username")),
                item.get("mal_id"),
                item.get("url"),
                json.dumps(item.get("images") or {}, ensure_ascii=True),
                now,
            )
            for item in users_payload
            if item.get("username")
        ]
        self.conn.executemany(
            """
            INSERT INTO users(run_id, username, mal_id, profile_url, images_json, discovered_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, username)
            DO UPDATE SET
                mal_id=COALESCE(excluded.mal_id, users.mal_id),
                profile_url=COALESCE(excluded.profile_url, users.profile_url),
                images_json=CASE
                    WHEN excluded.images_json != '{}' THEN excluded.images_json
                    ELSE users.images_json
                END
            """,
            rows,
        )
        self.conn.commit()
        after = self.count_users(run_id)
        return max(0, after - before)

    def count_users(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return int(row["c"])

    def count_user_anime_rows(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM user_anime_list WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return int(row["c"])

    def user_state_counts(self, run_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT crawl_state, COUNT(*) as c FROM users WHERE run_id=? GROUP BY crawl_state",
            (run_id,),
        ).fetchall()
        return {row["crawl_state"]: int(row["c"]) for row in rows}

    def get_usernames_for_ingestion(self, run_id: str, limit: int) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT username
            FROM users
            WHERE run_id=?
              AND crawl_state IN ('pending', 'error', 'rate_limited', 'not_found')
            ORDER BY discovered_at, username
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        return [str(row["username"]) for row in rows]

    def upsert_user_profile(self, run_id: str, profile: UserProfile) -> None:
        self.conn.execute(
            """
            INSERT INTO users(run_id, username, mal_id, profile_url, images_json, joined, last_online, is_public, discovered_at, crawl_state)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(run_id, username)
            DO UPDATE SET
                mal_id=COALESCE(excluded.mal_id, users.mal_id),
                profile_url=COALESCE(excluded.profile_url, users.profile_url),
                images_json=excluded.images_json,
                joined=COALESCE(excluded.joined, users.joined),
                last_online=COALESCE(excluded.last_online, users.last_online),
                is_public=excluded.is_public
            """,
            (
                run_id,
                profile.username,
                profile.mal_id,
                profile.profile_url,
                json.dumps(profile.images, ensure_ascii=True),
                profile.joined,
                profile.last_online,
                int(profile.is_public),
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def set_user_crawl_state(
        self,
        run_id: str,
        username: str,
        state: str,
        error_message: str | None = None,
    ) -> None:
        self.conn.execute(
            "UPDATE users SET crawl_state=?, crawl_error=? WHERE run_id=? AND username=?",
            (state, error_message, run_id, username),
        )
        self.conn.commit()

    def upsert_user_anime_entries(self, run_id: str, entries: list[UserAnimeEntry]) -> None:
        if not entries:
            return
        self.conn.executemany(
            """
            INSERT INTO user_anime_list(run_id, username, anime_mal_id, status, score, episodes_watched, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, username, anime_mal_id)
            DO UPDATE SET
                status=excluded.status,
                score=excluded.score,
                episodes_watched=excluded.episodes_watched,
                updated_at=excluded.updated_at
            """,
            [
                (
                    run_id,
                    entry.username,
                    entry.anime_mal_id,
                    entry.status,
                    entry.score,
                    entry.episodes_watched,
                    entry.updated_at,
                )
                for entry in entries
            ],
        )
        self.conn.commit()

    def upsert_user_reviews(self, run_id: str, username: str, reviews: list[dict[str, Any]]) -> None:
        if not reviews:
            return
        rows: list[tuple[Any, ...]] = []
        for review in reviews:
            entry = review.get("entry") or {}
            rows.append(
                (
                    run_id,
                    username,
                    review.get("mal_id"),
                    entry.get("mal_id"),
                    review.get("score"),
                    review.get("review"),
                    review.get("date") or review.get("created_at"),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO user_reviews(run_id, username, review_mal_id, anime_mal_id, score, review, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, username, review_mal_id)
            DO UPDATE SET
                anime_mal_id=excluded.anime_mal_id,
                score=excluded.score,
                review=excluded.review,
                created_at=excluded.created_at
            """,
            rows,
        )
        self.conn.commit()

    def enqueue_anime_ids(self, run_id: str, anime_ids: Iterable[int]) -> None:
        now = utc_now_iso()
        rows = [(run_id, int(anime_id), now) for anime_id in anime_ids]
        if not rows:
            return
        self.conn.executemany(
            """
            INSERT INTO anime_queue(run_id, anime_mal_id, frequency, processed, updated_at)
            VALUES(?, ?, 1, 0, ?)
            ON CONFLICT(run_id, anime_mal_id)
            DO UPDATE SET
                frequency=anime_queue.frequency + 1,
                updated_at=excluded.updated_at
            """,
            rows,
        )
        self.conn.commit()

    def get_pending_anime_ids(self, run_id: str) -> list[int]:
        rows = self.conn.execute(
            """
            SELECT anime_mal_id
            FROM anime_queue
            WHERE run_id=? AND processed=0
            ORDER BY frequency DESC, anime_mal_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [int(row["anime_mal_id"]) for row in rows]

    def count_anime_queue(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM anime_queue WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return int(row["c"])

    def count_anime_processed(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM anime_queue WHERE run_id=? AND processed=1",
            (run_id,),
        ).fetchone()
        return int(row["c"])

    def mark_anime_processed(
        self,
        run_id: str,
        anime_mal_id: int,
        state: str,
        error_message: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE anime_queue
            SET processed=1, last_state=?, error=?, updated_at=?
            WHERE run_id=? AND anime_mal_id=?
            """,
            (state, error_message, utc_now_iso(), run_id, anime_mal_id),
        )
        self.conn.commit()

    def upsert_anime_core(self, run_id: str, core: AnimeCore, nsfw: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO anime(
                run_id, mal_id, title, titles_json, type, source, episodes, status, duration, rating,
                synopsis, genres_json, themes_json, demographics_json, studios_json, producers_json,
                nsfw, fetched_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, mal_id)
            DO UPDATE SET
                title=excluded.title,
                titles_json=excluded.titles_json,
                type=excluded.type,
                source=excluded.source,
                episodes=excluded.episodes,
                status=excluded.status,
                duration=excluded.duration,
                rating=excluded.rating,
                synopsis=excluded.synopsis,
                genres_json=excluded.genres_json,
                themes_json=excluded.themes_json,
                demographics_json=excluded.demographics_json,
                studios_json=excluded.studios_json,
                producers_json=excluded.producers_json,
                nsfw=excluded.nsfw,
                fetched_at=excluded.fetched_at
            """,
            (
                run_id,
                core.mal_id,
                core.title,
                json.dumps(core.titles, ensure_ascii=True),
                core.type,
                core.source,
                core.episodes,
                core.status,
                core.duration,
                core.rating,
                core.synopsis,
                json.dumps(core.genres, ensure_ascii=True),
                json.dumps(core.themes, ensure_ascii=True),
                json.dumps(core.demographics, ensure_ascii=True),
                json.dumps(core.studios, ensure_ascii=True),
                json.dumps(core.producers, ensure_ascii=True),
                int(nsfw),
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def upsert_anime_stats(self, run_id: str, stats: AnimeStats) -> None:
        self.conn.execute(
            """
            INSERT INTO anime_statistics(
                run_id, mal_id, watching, completed, on_hold, dropped, plan_to_watch, total,
                scores_distribution_json, fetched_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, mal_id)
            DO UPDATE SET
                watching=excluded.watching,
                completed=excluded.completed,
                on_hold=excluded.on_hold,
                dropped=excluded.dropped,
                plan_to_watch=excluded.plan_to_watch,
                total=excluded.total,
                scores_distribution_json=excluded.scores_distribution_json,
                fetched_at=excluded.fetched_at
            """,
            (
                run_id,
                stats.mal_id,
                stats.watching,
                stats.completed,
                stats.on_hold,
                stats.dropped,
                stats.plan_to_watch,
                stats.total,
                json.dumps(stats.scores_distribution, ensure_ascii=True),
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def replace_anime_staff(self, run_id: str, anime_mal_id: int, staff: list[AnimeStaffEntry]) -> None:
        self.conn.execute(
            "DELETE FROM anime_staff WHERE run_id=? AND mal_id=?",
            (run_id, anime_mal_id),
        )
        if staff:
            self.conn.executemany(
                """
                INSERT INTO anime_staff(run_id, mal_id, person_mal_id, name, positions_json)
                VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        anime_mal_id,
                        row.person_mal_id,
                        row.name,
                        json.dumps(row.positions, ensure_ascii=True),
                    )
                    for row in staff
                ],
            )
        self.conn.commit()

    def upsert_anime_reviews(self, run_id: str, reviews: list[AnimeReviewEntry]) -> None:
        if not reviews:
            return
        self.conn.executemany(
            """
            INSERT INTO anime_reviews(
                run_id, review_key, review_mal_id, anime_mal_id, username, score, tags_json,
                is_spoiler, is_preliminary, review, reactions_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, review_key)
            DO UPDATE SET
                review_mal_id=excluded.review_mal_id,
                anime_mal_id=excluded.anime_mal_id,
                username=excluded.username,
                score=excluded.score,
                tags_json=excluded.tags_json,
                is_spoiler=excluded.is_spoiler,
                is_preliminary=excluded.is_preliminary,
                review=excluded.review,
                reactions_json=excluded.reactions_json,
                created_at=excluded.created_at
            """,
            [
                (
                    run_id,
                    review.review_key,
                    review.review_mal_id,
                    review.anime_mal_id,
                    review.username,
                    review.score,
                    json.dumps(review.tags, ensure_ascii=True),
                    int(review.is_spoiler),
                    int(review.is_preliminary),
                    review.review,
                    json.dumps(review.reactions, ensure_ascii=True),
                    review.created_at,
                )
                for review in reviews
            ],
        )
        self.conn.commit()

    def get_stats(self, run_id: str) -> RunStats:
        run = self.conn.execute(
            "SELECT run_id, target_users, status, started_at, finished_at FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"Unknown run id: {run_id}")

        user_states = self.user_state_counts(run_id)

        def count(query: str) -> int:
            row = self.conn.execute(query, (run_id,)).fetchone()
            return int(row["c"])

        return RunStats(
            run_id=run["run_id"],
            target_users=int(run["target_users"]),
            status=str(run["status"]),
            started_at=str(run["started_at"]),
            finished_at=run["finished_at"],
            users_discovered=count("SELECT COUNT(*) AS c FROM users WHERE run_id=?"),
            users_success=user_states.get("success", 0),
            users_private=user_states.get("private", 0),
            users_not_found=user_states.get("not_found", 0),
            users_errors=user_states.get("error", 0) + user_states.get("rate_limited", 0),
            user_anime_rows=count("SELECT COUNT(*) AS c FROM user_anime_list WHERE run_id=?"),
            completed_ratings_rows=count(
                """
                SELECT COUNT(*) AS c
                FROM user_anime_list
                WHERE run_id=? AND status='completed' AND score > 0
                """
            ),
            queued_anime=count("SELECT COUNT(*) AS c FROM anime_queue WHERE run_id=?"),
            processed_anime=count("SELECT COUNT(*) AS c FROM anime_queue WHERE run_id=? AND processed=1"),
            anime_rows=count("SELECT COUNT(*) AS c FROM anime WHERE run_id=?"),
            anime_stats_rows=count("SELECT COUNT(*) AS c FROM anime_statistics WHERE run_id=?"),
            anime_staff_rows=count("SELECT COUNT(*) AS c FROM anime_staff WHERE run_id=?"),
            anime_reviews_rows=count("SELECT COUNT(*) AS c FROM anime_reviews WHERE run_id=?"),
        )

    def latest_run_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return str(row["run_id"]) if row else None
