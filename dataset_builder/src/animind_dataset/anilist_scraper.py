from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import httpx
import pandas as pd
from rich.console import Console
from rich.progress import track

# Legacy timing rules preserved from the notebook implementation.
CLUB_PAGE_DELAY_SECONDS = 3.0
API_DELAY_SECONDS = 4.2
RETRY_COOLDOWN_SECONDS = 120.0
KAGGLE_DEFAULT_DATASET = "hernan4444/anime-recommendation-database-2020"

STATUS_TO_ID = {
    "watching": 1,
    "completed": 2,
    "on hold": 3,
    "dropped": 4,
    "plan to watch": 6,
}
TERMINAL_CLUB_STATUSES = {403, 404}
TERMINAL_USER_STATUSES = {400, 403, 404}
TERMINAL_ANIME_STATUSES = {400, 404}
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
PhaseName = Literal["all", "clubs", "users", "animelist", "anime", "export"]


@dataclass(slots=True)
class ScrapeConfig:
    output_dir: Path
    target_possible_users: int = 1_000_000
    min_club_members: int = 30
    max_club_pages: int = 0
    timeout_seconds: float = 20.0
    club_workers: int = 1
    club_limit: int = 0
    club_page_delay_seconds: float = CLUB_PAGE_DELAY_SECONDS
    api_delay_seconds: float = API_DELAY_SECONDS
    jikan_base_url: str = "https://api.jikan.moe/v4"

    @property
    def sqlite_path(self) -> Path:
        return self.output_dir / "anilist.sqlite"


class DelayGate:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._last_tick = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self.delay_seconds - (now - self._last_tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_tick = time.monotonic()


class AniListScraper:
    def __init__(self, config: ScrapeConfig, console: Console) -> None:
        self.config = config
        self.console = console
        self.base_url = self.config.jikan_base_url.rstrip("/")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.config.sqlite_path)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()
        self.client = httpx.Client(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "animind-dataset/0.1"},
        )
        self.club_gate = DelayGate(self.config.club_page_delay_seconds)
        self.api_gate = DelayGate(self.config.api_delay_seconds)

    def close(self) -> None:
        self.client.close()
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS clubs (
                club_id INTEGER PRIMARY KEY,
                members INTEGER NOT NULL,
                discovered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS club_scan_progress (
                club_id INTEGER PRIMARY KEY,
                scanned_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                source_club_id INTEGER,
                discovered_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_anime (
                user_id INTEGER NOT NULL,
                anime_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                watching_status INTEGER NOT NULL,
                watched_episodes INTEGER NOT NULL,
                scraped_at TEXT NOT NULL,
                PRIMARY KEY (user_id, anime_id),
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );
            CREATE TABLE IF NOT EXISTS anime (
                anime_id INTEGER PRIMARY KEY,
                name TEXT,
                english_name TEXT,
                score REAL,
                genres TEXT,
                synopsis TEXT,
                type TEXT,
                episodes INTEGER,
                premiered TEXT,
                studios TEXT,
                source TEXT,
                rating TEXT,
                members INTEGER,
                favorites INTEGER,
                scored_by INTEGER,
                imported_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_anime_anime_id
            ON user_anime(anime_id);
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS club_member_progress (
                club_id INTEGER PRIMARY KEY,
                last_page INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (club_id) REFERENCES clubs (club_id)
            );
            CREATE TABLE IF NOT EXISTS user_animelist_progress (
                user_id INTEGER PRIMARY KEY,
                last_page INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );
            CREATE TABLE IF NOT EXISTS anime_catalog_progress (
                anime_id INTEGER PRIMARY KEY,
                last_status INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_club_member_progress_completed
            ON club_member_progress(completed);
            CREATE INDEX IF NOT EXISTS idx_user_animelist_progress_completed
            ON user_animelist_progress(completed);
            CREATE INDEX IF NOT EXISTS idx_anime_catalog_progress_completed
            ON anime_catalog_progress(completed);
            """
        )
        # Backward-compatible migration from old completion-only progress.
        self.conn.execute(
            """
            INSERT OR REPLACE INTO club_member_progress(club_id, last_page, completed, updated_at)
            SELECT club_id, 0, 1, scanned_at
            FROM club_scan_progress
            """
        )
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    def _api_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def _set_states(self, values: dict[str, str]) -> None:
        self.conn.executemany(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            list(values.items()),
        )
        self.conn.commit()

    def _get_state_int(self, key: str, default: int = 0) -> int:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return int(row[0])
        except ValueError:
            return default

    def _upsert_club_progress(self, club_id: int, last_page: int, completed: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO club_member_progress(club_id, last_page, completed, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(club_id) DO UPDATE SET
                last_page=excluded.last_page,
                completed=excluded.completed,
                updated_at=excluded.updated_at
            """,
            (club_id, max(0, last_page), int(completed), self._now()),
        )

    def _upsert_user_progress(self, user_id: int, last_page: int, completed: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO user_animelist_progress(user_id, last_page, completed, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_page=excluded.last_page,
                completed=excluded.completed,
                updated_at=excluded.updated_at
            """,
            (user_id, max(0, last_page), int(completed), self._now()),
        )

    def _upsert_anime_progress(self, anime_id: int, status: int, completed: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO anime_catalog_progress(anime_id, last_status, completed, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(anime_id) DO UPDATE SET
                last_status=excluded.last_status,
                completed=excluded.completed,
                updated_at=excluded.updated_at
            """,
            (anime_id, int(status), int(completed), self._now()),
        )

    def resume_snapshot(self) -> dict[str, int]:
        clubs_total = self.conn.execute("SELECT COUNT(*) FROM clubs").fetchone()[0]
        clubs_done = self.conn.execute(
            "SELECT COUNT(*) FROM club_member_progress WHERE completed = 1"
        ).fetchone()[0]
        users_total = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        users_done = self.conn.execute(
            "SELECT COUNT(*) FROM user_animelist_progress WHERE completed = 1"
        ).fetchone()[0]
        user_anime_rows = self.conn.execute("SELECT COUNT(*) FROM user_anime").fetchone()[0]
        anime_rows = self.conn.execute("SELECT COUNT(*) FROM anime").fetchone()[0]
        return {
            "clubs_total": int(clubs_total),
            "clubs_done": int(clubs_done),
            "users_total": int(users_total),
            "users_done": int(users_done),
            "user_anime_rows": int(user_anime_rows),
            "anime_rows": int(anime_rows),
        }

    def log_resume_status(self) -> None:
        stats = self.resume_snapshot()
        clubs_pct = (
            (100.0 * stats["clubs_done"] / stats["clubs_total"]) if stats["clubs_total"] else 0.0
        )
        users_pct = (
            (100.0 * stats["users_done"] / stats["users_total"]) if stats["users_total"] else 0.0
        )
        self.console.log(
            "Resume status: "
            f"clubs {stats['clubs_done']:,}/{stats['clubs_total']:,} ({clubs_pct:.1f}%), "
            f"users {stats['users_done']:,}/{stats['users_total']:,} ({users_pct:.1f}%), "
            f"user_anime rows {stats['user_anime_rows']:,}, anime rows {stats['anime_rows']:,}"
        )

    def reset_resume_state(self) -> None:
        self.conn.execute("DELETE FROM club_member_progress")
        self.conn.execute("DELETE FROM user_animelist_progress")
        self.conn.execute("DELETE FROM anime_catalog_progress")
        self.conn.execute("DELETE FROM club_scan_progress")
        self.conn.execute(
            "DELETE FROM state WHERE key IN ('last_user_id', 'phase1_next_page', 'phase1_possible_users', 'phase1_done')"
        )
        self.conn.commit()

    def bootstrap_from_kaggle(
        self,
        dataset_handle: str = KAGGLE_DEFAULT_DATASET,
        force_download: bool = False,
    ) -> Path:
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub is not installed. Run `uv add kagglehub` in dataset_builder."
            ) from exc

        self.console.rule("[bold]Bootstrap - Kaggle dataset")
        if force_download:
            dataset_path = Path(kagglehub.dataset_download(dataset_handle, force_download=True))
        else:
            dataset_path = Path(kagglehub.dataset_download(dataset_handle))
        self.console.log(f"Downloaded dataset to: {dataset_path}")
        self.bootstrap_from_directory(dataset_path)
        return dataset_path

    def bootstrap_from_directory(self, dataset_path: Path) -> None:
        root = dataset_path if dataset_path.is_dir() else dataset_path.parent
        csv_files = list(root.rglob("*.csv"))
        self.console.log(f"Bootstrap scan: found {len(csv_files)} CSV files in {root}")

        anime_loaded = 0
        interactions_loaded = 0

        anime_synopsis = self._find_csv(root, "anime_with_synopsis.csv")
        if anime_synopsis is not None:
            anime_loaded += self._import_anime_csv(anime_synopsis)
        anime_csv = self._find_csv(root, "anime.csv")
        if anime_csv is not None:
            anime_loaded += self._import_anime_csv(anime_csv)

        animelist_csv = self._find_csv(root, "animelist.csv")
        if animelist_csv is not None:
            interactions_loaded += self._import_user_anime_csv(animelist_csv, default_status=2)

        rating_complete_csv = self._find_csv(root, "rating_complete.csv")
        if rating_complete_csv is not None:
            interactions_loaded += self._import_user_anime_csv(rating_complete_csv, default_status=2)

        rating_csv = self._find_csv(root, "rating.csv")
        if rating_csv is not None:
            interactions_loaded += self._import_user_anime_csv(rating_csv, default_status=2)

        self.console.log(
            f"[green]Bootstrap complete[/green]: "
            f"{anime_loaded:,} anime rows imported, {interactions_loaded:,} interaction rows imported."
        )

    def _find_csv(self, root: Path, filename: str) -> Path | None:
        filename_lower = filename.lower()
        for path in root.rglob("*.csv"):
            if path.name.lower() == filename_lower:
                return path
        return None

    @staticmethod
    def _normalize_columns(columns: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for raw in columns:
            normalized = raw.strip().lower().replace(" ", "_")
            mapping[normalized] = raw
        return mapping

    @staticmethod
    def _find_normalized_column(normalized_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in normalized_map:
                return normalized_map[candidate]
        return None

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_text(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _join_names(value: Any) -> str | None:
        if not isinstance(value, list):
            return None
        names = [str(item.get("name", "")).strip() for item in value if isinstance(item, dict)]
        names = [name for name in names if name]
        return ", ".join(dict.fromkeys(names)) if names else None

    def _as_status(self, value: Any, default: int) -> int:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            key = value.strip().lower().replace("-", " ").replace("_", " ")
            return STATUS_TO_ID.get(key, default)
        parsed = self._as_int(value, default=default)
        return parsed if parsed >= 0 else default

    def _iter_csv_chunks(self, file_path: Path, chunksize: int = 50_000):
        return pd.read_csv(file_path, chunksize=chunksize, low_memory=True)

    def _import_anime_csv(self, file_path: Path) -> int:
        self.console.log(f"Importing anime data from {file_path.name}")
        inserted = 0
        for chunk in self._iter_csv_chunks(file_path, chunksize=10_000):
            normalized_map = self._normalize_columns(list(chunk.columns))
            id_col = self._find_normalized_column(normalized_map, ("anime_id", "mal_id"))
            if id_col is None:
                continue

            name_col = self._find_normalized_column(normalized_map, ("name", "title"))
            english_name_col = self._find_normalized_column(
                normalized_map, ("english_name", "title_english")
            )
            score_col = self._find_normalized_column(normalized_map, ("score",))
            genres_col = self._find_normalized_column(normalized_map, ("genres",))
            synopsis_col = self._find_normalized_column(normalized_map, ("synopsis",))
            type_col = self._find_normalized_column(normalized_map, ("type",))
            episodes_col = self._find_normalized_column(normalized_map, ("episodes",))
            premiered_col = self._find_normalized_column(normalized_map, ("premiered",))
            studios_col = self._find_normalized_column(normalized_map, ("studios",))
            source_col = self._find_normalized_column(normalized_map, ("source",))
            rating_col = self._find_normalized_column(normalized_map, ("rating",))
            members_col = self._find_normalized_column(normalized_map, ("members",))
            favorites_col = self._find_normalized_column(normalized_map, ("favorites",))
            scored_by_col = self._find_normalized_column(normalized_map, ("scored_by",))

            rows: list[tuple[Any, ...]] = []
            now = self._now()
            for row in chunk.itertuples(index=False):
                row_map = dict(zip(chunk.columns, row, strict=False))
                anime_id = self._as_int(row_map.get(id_col), default=0)
                if anime_id <= 0:
                    continue
                rows.append(
                    (
                        anime_id,
                        self._as_text(row_map.get(name_col)) if name_col else None,
                        self._as_text(row_map.get(english_name_col)) if english_name_col else None,
                        self._as_float(row_map.get(score_col)) if score_col else None,
                        self._as_text(row_map.get(genres_col)) if genres_col else None,
                        self._as_text(row_map.get(synopsis_col)) if synopsis_col else None,
                        self._as_text(row_map.get(type_col)) if type_col else None,
                        self._as_int(row_map.get(episodes_col), default=0) if episodes_col else None,
                        self._as_text(row_map.get(premiered_col)) if premiered_col else None,
                        self._as_text(row_map.get(studios_col)) if studios_col else None,
                        self._as_text(row_map.get(source_col)) if source_col else None,
                        self._as_text(row_map.get(rating_col)) if rating_col else None,
                        self._as_int(row_map.get(members_col), default=0) if members_col else None,
                        self._as_int(row_map.get(favorites_col), default=0) if favorites_col else None,
                        self._as_int(row_map.get(scored_by_col), default=0) if scored_by_col else None,
                        now,
                    )
                )

            if not rows:
                continue

            before = self.conn.total_changes
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO anime(
                    anime_id, name, english_name, score, genres, synopsis, type,
                    episodes, premiered, studios, source, rating, members,
                    favorites, scored_by, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()
            inserted += self.conn.total_changes - before

        self.console.log(f"Anime import from {file_path.name}: +{inserted:,} rows")
        return inserted

    def _import_user_anime_csv(self, file_path: Path, default_status: int) -> int:
        self.console.log(f"Importing interactions from {file_path.name}")
        inserted = 0
        for chunk in self._iter_csv_chunks(file_path):
            normalized_map = self._normalize_columns(list(chunk.columns))
            user_col = self._find_normalized_column(normalized_map, ("user_id", "userid"))
            anime_col = self._find_normalized_column(normalized_map, ("anime_id", "mal_id", "animeid"))
            score_col = self._find_normalized_column(normalized_map, ("rating", "score"))
            status_col = self._find_normalized_column(normalized_map, ("watching_status", "status"))
            episodes_col = self._find_normalized_column(
                normalized_map, ("watched_episodes", "episodes_watched")
            )
            if user_col is None or anime_col is None:
                continue

            now = self._now()
            user_ids: set[int] = set()
            interaction_rows: list[tuple[int, int, int, int, int, str]] = []

            for row in chunk.itertuples(index=False):
                row_map = dict(zip(chunk.columns, row, strict=False))
                user_id = self._as_int(row_map.get(user_col), default=0)
                anime_id = self._as_int(row_map.get(anime_col), default=0)
                if user_id <= 0 or anime_id <= 0:
                    continue

                score = self._as_int(row_map.get(score_col), default=0) if score_col else 0
                if score < 0:
                    score = 0
                status = self._as_status(row_map.get(status_col), default=default_status) if status_col else default_status
                watched_episodes = (
                    self._as_int(row_map.get(episodes_col), default=0) if episodes_col else 0
                )
                user_ids.add(user_id)
                interaction_rows.append((user_id, anime_id, score, status, watched_episodes, now))

            if not interaction_rows:
                continue

            self.conn.executemany(
                """
                INSERT OR IGNORE INTO users(user_id, username, source_club_id, discovered_at)
                VALUES (?, ?, NULL, ?)
                """,
                [(user_id, f"kaggle_user_{user_id}", now) for user_id in sorted(user_ids)],
            )
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO user_animelist_progress(user_id, last_page, completed, updated_at)
                VALUES (?, 0, 1, ?)
                """,
                [(user_id, now) for user_id in sorted(user_ids)],
            )
            before = self.conn.total_changes
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO user_anime(
                    user_id, anime_id, score, watching_status, watched_episodes, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                interaction_rows,
            )
            self.conn.commit()
            inserted += self.conn.total_changes - before

        self.console.log(f"Interactions import from {file_path.name}: +{inserted:,} rows")
        return inserted

    def _json_request(self, url: str, gate: DelayGate) -> tuple[int, dict[str, Any]]:
        while True:
            gate.wait()
            try:
                response = self.client.get(url)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.console.log(f"[yellow]Request failed[/yellow] {url} ({exc}). Cooling down 120s.")
                time.sleep(RETRY_COOLDOWN_SECONDS)
                continue

            if response.status_code in RETRYABLE_STATUSES:
                self.console.log(
                    f"[yellow]HTTP {response.status_code}[/yellow] for {url}. Cooling down 120s."
                )
                time.sleep(RETRY_COOLDOWN_SECONDS)
                continue

            if response.status_code != 200:
                return response.status_code, {}

            try:
                return 200, response.json()
            except ValueError:
                self.console.log(f"[yellow]Bad JSON[/yellow] from {url}. Cooling down 120s.")
                time.sleep(RETRY_COOLDOWN_SECONDS)

    def discover_clubs(self) -> int:
        self.console.rule("[bold]Phase 1/4 - Discover clubs")
        phase_complete = False
        known_possible = self._get_state_int("phase1_possible_users", default=-1)
        total_possible_users = (
            int(self.conn.execute("SELECT COALESCE(SUM(members), 0) FROM clubs").fetchone()[0])
            if known_possible < 0
            else known_possible
        )
        page = max(1, self._get_state_int("phase1_next_page", default=1))
        seen_ids = {
            row[0] for row in self.conn.execute("SELECT club_id FROM clubs").fetchall()
        }

        while True:
            if self.config.max_club_pages > 0 and page > self.config.max_club_pages:
                break

            status, payload = self._json_request(
                self._api_url(f"clubs?page={page}"),
                self.club_gate,
            )
            if status != 200:
                self.console.log(f"[yellow]Stopped club discovery at page {page} (HTTP {status}).[/yellow]")
                break

            clubs = payload.get("data", [])
            if not clubs:
                phase_complete = True
                break

            added_this_page = 0
            now = self._now()
            for item in clubs:
                club_id = int(item.get("mal_id", 0))
                members = int(item.get("members", 0))
                if club_id <= 0 or members <= self.config.min_club_members:
                    continue
                if club_id in seen_ids:
                    continue
                self.conn.execute(
                    "INSERT OR IGNORE INTO clubs(club_id, members, discovered_at) VALUES(?, ?, ?)",
                    (club_id, members, now),
                )
                seen_ids.add(club_id)
                total_possible_users += members
                added_this_page += 1

            self.conn.commit()
            self._set_states(
                {
                    "phase1_next_page": str(page + 1),
                    "phase1_possible_users": str(total_possible_users),
                    "phase1_done": "0",
                }
            )
            self.console.log(
                f"Page {page}: +{added_this_page} clubs "
                f"(possible users progress: {total_possible_users:,}/{self.config.target_possible_users:,})"
            )

            has_next = bool(payload.get("pagination", {}).get("has_next_page"))
            if total_possible_users >= self.config.target_possible_users or not has_next:
                phase_complete = True
                break
            page += 1

        self._set_state("phase1_done", "1" if phase_complete else "0")
        count = self.conn.execute("SELECT COUNT(*) FROM clubs").fetchone()[0]
        self.console.log(f"[green]Clubs ready:[/green] {count:,}")
        return count

    def collect_users_from_clubs(self) -> int:
        self.console.rule("[bold]Phase 2/4 - Collect users")
        club_rows = self.conn.execute(
            """
            SELECT c.club_id, COALESCE(p.last_page, 0), COALESCE(p.completed, 0)
            FROM clubs c
            LEFT JOIN club_member_progress p ON p.club_id = c.club_id
            ORDER BY c.club_id
            """
        ).fetchall()
        pending = sum(1 for _, _, completed in club_rows if completed == 0)
        self.console.log(f"Phase 2 resume: {pending:,} clubs pending.")
        pending_rows = [(club_id, last_page) for club_id, last_page, completed in club_rows if completed == 0]
        if self.config.club_limit > 0:
            pending_rows = pending_rows[: self.config.club_limit]
        if not pending_rows:
            count = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            self.console.log(f"[green]Users ready:[/green] {count:,}")
            return count

        workers = max(1, int(self.config.club_workers))
        self.console.log(
            f"Phase 2 workers: {workers} (api delay {self.config.api_delay_seconds:.3f}s, base {self.base_url})"
        )
        started_at = time.monotonic()
        processed = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._fetch_single_club, club_id, int(last_page)) for club_id, last_page in pending_rows]
            for future in track(as_completed(futures), total=len(futures), description="Clubs"):
                club_id, checkpoint_page, completed, usernames, message = future.result()
                processed += 1
                now = self._now()
                if usernames:
                    self.conn.executemany(
                        "INSERT OR IGNORE INTO users(username, source_club_id, discovered_at) VALUES(?, ?, ?)",
                        [(username, club_id, now) for username in usernames],
                    )
                self._upsert_club_progress(club_id, checkpoint_page, completed)
                if completed:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO club_scan_progress(club_id, scanned_at) VALUES(?, ?)",
                        (club_id, now),
                    )
                self.conn.commit()
                if message:
                    self.console.log(message)

        elapsed = max(0.001, time.monotonic() - started_at)
        self.console.log(
            f"Phase 2 throughput: {processed / elapsed:.2f} clubs/sec over {processed} clubs ({elapsed:.1f}s)."
        )
        count = self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.console.log(f"[green]Users ready:[/green] {count:,}")
        return count

    def _fetch_single_club(
        self,
        club_id: int,
        last_page: int,
    ) -> tuple[int, int, bool, list[str], str | None]:
        page = max(1, int(last_page) + 1)
        usernames: set[str] = set()
        while True:
            status, payload = self._json_request(
                self._api_url(f"clubs/{club_id}/members?page={page}"),
                self.api_gate,
            )
            if status != 200:
                checkpoint_page = max(int(last_page), page - 1)
                if status in TERMINAL_CLUB_STATUSES:
                    return (
                        club_id,
                        checkpoint_page,
                        True,
                        sorted(usernames),
                        f"[yellow]Club {club_id} returned HTTP {status}; checkpointing as skipped.[/yellow]",
                    )
                return (
                    club_id,
                    checkpoint_page,
                    False,
                    sorted(usernames),
                    f"[yellow]Club {club_id} paused at page {checkpoint_page}; will resume later.[/yellow]",
                )

            members = payload.get("data", [])
            if not members:
                return club_id, page, True, sorted(usernames), None

            for member in members:
                username = self._extract_username(member)
                if username:
                    usernames.add(username)

            has_next = bool(payload.get("pagination", {}).get("has_next_page"))
            if not has_next:
                return club_id, page, True, sorted(usernames), None
            page += 1

    def collect_user_animelist(self) -> int:
        self.console.rule("[bold]Phase 3/4 - Collect user anime lists")
        user_rows = self.conn.execute(
            """
            SELECT u.user_id, u.username, COALESCE(p.last_page, 0), COALESCE(p.completed, 0)
            FROM users u
            LEFT JOIN user_animelist_progress p ON p.user_id = u.user_id
            ORDER BY u.user_id
            """
        ).fetchall()
        pending = sum(1 for _, _, _, completed in user_rows if completed == 0)
        self.console.log(f"Phase 3 resume: {pending:,} users pending.")

        for user_id, username, last_page, completed in track(user_rows, description="Users"):
            if completed == 1:
                continue
            page = max(1, int(last_page) + 1)
            while True:
                status, payload = self._json_request(
                    self._api_url(f"users/{username}/animelist?status=all&page={page}"),
                    self.api_gate,
                )
                if status != 200:
                    checkpoint_page = max(int(last_page), page - 1)
                    if status in TERMINAL_USER_STATUSES:
                        self.console.log(
                            f"[yellow]User {username} returned HTTP {status}; checkpointing as skipped.[/yellow]"
                        )
                        self._upsert_user_progress(user_id, checkpoint_page, True)
                        self.conn.commit()
                        self._set_state("last_user_id", str(user_id))
                    else:
                        self._upsert_user_progress(user_id, checkpoint_page, False)
                        self.conn.commit()
                        self.console.log(
                            f"[yellow]User {username} paused at page {checkpoint_page}; will resume later.[/yellow]"
                        )
                    break

                items = payload.get("data", [])
                if not items:
                    self._upsert_user_progress(user_id, page, True)
                    self.conn.commit()
                    self._set_state("last_user_id", str(user_id))
                    break

                now = self._now()
                rows: list[tuple[int, int, int, int, int, str]] = []
                for item in items:
                    anime_id = self._extract_anime_id(item)
                    if anime_id <= 0:
                        continue
                    rows.append(
                        (
                            user_id,
                            anime_id,
                            self._extract_score(item),
                            self._extract_status(item),
                            self._extract_watched_episodes(item),
                            now,
                        )
                    )

                self.conn.executemany(
                    """
                    INSERT INTO user_anime(
                        user_id, anime_id, score, watching_status, watched_episodes, scraped_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, anime_id) DO UPDATE SET
                        score=excluded.score,
                        watching_status=excluded.watching_status,
                        watched_episodes=excluded.watched_episodes,
                        scraped_at=excluded.scraped_at
                    """,
                    rows,
                )
                self._upsert_user_progress(user_id, page, False)
                self.conn.commit()

                has_next = bool(payload.get("pagination", {}).get("has_next_page"))
                if not has_next:
                    self._upsert_user_progress(user_id, page, True)
                    self.conn.commit()
                    self._set_state("last_user_id", str(user_id))
                    break
                page += 1

        count = self.conn.execute("SELECT COUNT(*) FROM user_anime").fetchone()[0]
        self.console.log(f"[green]User anime rows:[/green] {count:,}")
        return count

    def collect_anime_catalog(self) -> int:
        self.console.rule("[bold]Phase 4/4 - Build anime catalog")
        missing_rows = self.conn.execute(
            """
            SELECT DISTINCT ua.anime_id
            FROM user_anime ua
            LEFT JOIN anime a ON a.anime_id = ua.anime_id
            LEFT JOIN anime_catalog_progress p ON p.anime_id = ua.anime_id
            WHERE a.anime_id IS NULL AND COALESCE(p.completed, 0) = 0
            ORDER BY ua.anime_id
            """
        ).fetchall()
        missing_ids = [int(row[0]) for row in missing_rows]
        self.console.log(f"Phase 4 resume: {len(missing_ids):,} anime pending.")

        for anime_id in track(missing_ids, description="Anime"):
            status, payload = self._json_request(self._api_url(f"anime/{anime_id}/full"), self.api_gate)
            if status != 200:
                completed = status in TERMINAL_ANIME_STATUSES
                self._upsert_anime_progress(anime_id, status, completed)
                self.conn.commit()
                if completed:
                    self.console.log(
                        f"[yellow]Anime {anime_id} returned HTTP {status}; checkpointing as skipped.[/yellow]"
                    )
                continue

            data = payload.get("data")
            if not isinstance(data, dict):
                self._upsert_anime_progress(anime_id, 0, False)
                self.conn.commit()
                continue

            row = self._anime_row_from_payload(data)
            if row is None:
                self._upsert_anime_progress(anime_id, 0, False)
                self.conn.commit()
                continue

            self.conn.execute(
                """
                INSERT OR IGNORE INTO anime(
                    anime_id, name, english_name, score, genres, synopsis, type,
                    episodes, premiered, studios, source, rating, members,
                    favorites, scored_by, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._upsert_anime_progress(anime_id, 200, True)
            self.conn.commit()

        count = self.conn.execute("SELECT COUNT(*) FROM anime").fetchone()[0]
        self.console.log(f"[green]Anime catalog rows:[/green] {count:,}")
        return count

    def _anime_row_from_payload(self, data: dict[str, Any]) -> tuple[Any, ...] | None:
        anime_id = self._as_int(data.get("mal_id"), default=0)
        if anime_id <= 0:
            return None
        genres = self._join_names(data.get("genres"))
        themes = self._join_names(data.get("themes"))
        demographics = self._join_names(data.get("demographics"))
        genre_parts = [text for text in (genres, themes, demographics) if text]
        premiered = None
        season = self._as_text(data.get("season"))
        year = self._as_int(data.get("year"), default=0)
        if season and year > 0:
            premiered = f"{season} {year}"
        elif season:
            premiered = season
        elif year > 0:
            premiered = str(year)
        return (
            anime_id,
            self._as_text(data.get("title")),
            self._as_text(data.get("title_english")),
            self._as_float(data.get("score")),
            ", ".join(genre_parts) if genre_parts else None,
            self._as_text(data.get("synopsis")),
            self._as_text(data.get("type")),
            self._as_int(data.get("episodes"), default=0) or None,
            premiered,
            self._join_names(data.get("studios")),
            self._as_text(data.get("source")),
            self._as_text(data.get("rating")),
            self._as_int(data.get("members"), default=0) or None,
            self._as_int(data.get("favorites"), default=0) or None,
            self._as_int(data.get("scored_by"), default=0) or None,
            self._now(),
        )

    def export_parquet(self) -> None:
        self.console.rule("[bold]Export - Parquet")
        for table in (
            "clubs",
            "users",
            "user_anime",
            "anime",
            "club_member_progress",
            "user_animelist_progress",
            "anime_catalog_progress",
            "state",
        ):
            df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
            out_path = self.config.output_dir / f"{table}.parquet"
            df.to_parquet(out_path, index=False)
            self.console.log(f"[green]Wrote[/green] {out_path}")

    @staticmethod
    def _extract_username(member: dict[str, Any]) -> str:
        if isinstance(member.get("username"), str):
            return member["username"].strip()
        user_data = member.get("user")
        if isinstance(user_data, dict) and isinstance(user_data.get("username"), str):
            return user_data["username"].strip()
        return ""

    @staticmethod
    def _extract_anime_id(item: dict[str, Any]) -> int:
        anime = item.get("anime")
        if isinstance(anime, dict):
            return int(anime.get("mal_id") or 0)
        return int(item.get("anime_id") or item.get("mal_id") or 0)

    @staticmethod
    def _extract_score(item: dict[str, Any]) -> int:
        value = item.get("score", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_status(item: dict[str, Any]) -> int:
        status = item.get("status")
        if isinstance(status, int):
            return status
        if isinstance(status, str):
            key = status.strip().lower().replace("-", " ").replace("_", " ")
            return STATUS_TO_ID.get(key, 0)
        return 0

    @staticmethod
    def _extract_watched_episodes(item: dict[str, Any]) -> int:
        for key in ("episodes_watched", "watched_episodes"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0


def run_scrape(
    config: ScrapeConfig,
    export_only: bool = False,
    phase: PhaseName = "all",
    status_only: bool = False,
    reset_resume: bool = False,
    bootstrap_kaggle: bool = False,
    kaggle_dataset: str = KAGGLE_DEFAULT_DATASET,
    kaggle_force_download: bool = False,
    bootstrap_path: Path | None = None,
) -> None:
    console = Console()
    scraper = AniListScraper(config=config, console=console)
    try:
        if reset_resume:
            scraper.reset_resume_state()
            console.log("[yellow]Resume checkpoints cleared.[/yellow]")

        if bootstrap_kaggle:
            if bootstrap_path is not None:
                scraper.console.rule("[bold]Bootstrap - Local dataset path")
                scraper.bootstrap_from_directory(bootstrap_path)
            else:
                scraper.bootstrap_from_kaggle(
                    dataset_handle=kaggle_dataset,
                    force_download=kaggle_force_download,
                )

        scraper.log_resume_status()
        if status_only:
            return

        if export_only:
            phase = "export"

        if phase in ("all", "clubs"):
            scraper.discover_clubs()
        if phase in ("all", "users"):
            scraper.collect_users_from_clubs()
        if phase in ("all", "animelist"):
            scraper.collect_user_animelist()
        if phase in ("all", "anime"):
            scraper.collect_anime_catalog()
        if phase in ("all", "export"):
            scraper.export_parquet()

        scraper.log_resume_status()
        console.log(f"[bold green]Done[/bold green]. SQLite: {config.sqlite_path}")
    finally:
        scraper.close()
