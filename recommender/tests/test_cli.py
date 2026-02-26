from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from animind_recommender.cli import app


def _make_source_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE anime (
                anime_id INTEGER PRIMARY KEY,
                name TEXT,
                english_name TEXT,
                genres TEXT,
                synopsis TEXT,
                studios TEXT,
                year INTEGER,
                premiered TEXT,
                type TEXT,
                source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE user_anime (
                user_id INTEGER NOT NULL,
                anime_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                watching_status INTEGER NOT NULL,
                watched_episodes INTEGER NOT NULL,
                scraped_at TEXT NOT NULL,
                PRIMARY KEY(user_id, anime_id)
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO anime(
                anime_id, name, english_name, genres, synopsis, studios, year, premiered, type, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "A", "A", "Action, Drama", "s1", "StudioA", 2001, "Spring 2001", "TV", "Original"),
                (2, "B", "B", "Drama, Sci-Fi", "s2", "StudioB", 2002, "Spring 2002", "TV", "Manga"),
                (3, "C", "C", "Sci-Fi, Mystery", "s3", "StudioC", 2003, "Spring 2003", "TV", "Novel"),
                (4, "D", "D", "Action, Sci-Fi", "s4", "StudioD", 2004, "Spring 2004", "TV", "Original"),
            ],
        )
        rows = []
        for user_id in range(1, 5):
            rows.extend(
                [
                    (user_id, 1, 8, 2, 12, "2026-02-20T00:00:00+00:00"),
                    (user_id, 2, 8, 2, 12, "2026-02-20T00:00:00+00:00"),
                    (user_id, 3, 7, 2, 12, "2026-02-20T00:00:00+00:00"),
                    (user_id, 4, 6, 1, 4, "2026-02-20T00:00:00+00:00"),
                ]
            )
        conn.executemany(
            """
            INSERT INTO user_anime(user_id, anime_id, score, watching_status, watched_episodes, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _make_semantic_ids(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for anime_id in range(1, 5):
            row = {
                "anime_id": anime_id,
                "name": f"N{anime_id}",
                "english_name": f"EN{anime_id}",
                "genres": "Action, Drama",
                "semantic_id": "".join(f"<L{i}_{anime_id}>" for i in range(1, 9)),
                "tokens": [f"<L{i}_{anime_id}>" for i in range(1, 9)],
                "codes": [anime_id] * 8,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_cli_run_phase_prep_smoke(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    semantic_ids = tmp_path / "semantic_ids.jsonl"
    out_dir = tmp_path / "out"
    _make_source_db(source_db)
    _make_semantic_ids(semantic_ids)

    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                f'source_db = "{source_db}"',
                f'semantic_ids_path = "{semantic_ids}"',
                f'out_dir = "{out_dir}"',
                "rebuild = true",
                "seed = 7",
                "target_examples = 12",
                "split_task_a = 0.5",
                "split_task_b = 0.25",
                "split_task_c = 0.25",
                "max_task_a_templates_per_anime = 8",
                "task_b_min_history = 4",
                "task_b_max_history = 10",
                'task_b_mask_modes = ["last", "random"]',
                "task_b_positive_score_min = 7",
                "task_b_allowed_statuses = [1,2,3,4,6]",
                "export_parquet = false",
                "write_manifest = true",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--phase",
            "prep",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "llm_prep_train.jsonl").exists()
    assert (out_dir / "llm_prep_summary.json").exists()
