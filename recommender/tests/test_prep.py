from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from animind_recommender.prep import PrepConfig, run_prep


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

        anime_rows = [
            (1, "Cowboy Bebop", "Cowboy Bebop", "Action, Sci-Fi, Drama", "Space bounty hunters", "Sunrise", 1998, "Spring 1998", "TV", "Original"),
            (2, "Neon Genesis Evangelion", "Neon Genesis Evangelion", "Drama, Sci-Fi, Psychological", "Pilots fight angels", "Gainax", 1995, "Fall 1995", "TV", "Original"),
            (3, "Psycho-Pass", "Psycho-Pass", "Sci-Fi, Psychological, Thriller", "Crime in dystopia", "Production I.G", 2012, "Fall 2012", "TV", "Original"),
            (4, "Serial Experiments Lain", "Serial Experiments Lain", "Psychological, Mystery, Sci-Fi", "Wired consciousness", "Triangle Staff", 1998, "Summer 1998", "TV", "Original"),
            (5, "Trigun", "Trigun", "Action, Adventure, Sci-Fi", "Desert gunslinger", "Madhouse", 1998, "Spring 1998", "TV", "Manga"),
            (6, "Samurai Champloo", "Samurai Champloo", "Action, Adventure", "Samurai road trip", "Manglobe", 2004, "Spring 2004", "TV", "Original"),
        ]
        conn.executemany(
            """
            INSERT INTO anime(
                anime_id, name, english_name, genres, synopsis, studios, year, premiered, type, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            anime_rows,
        )

        user_anime_rows = []
        for user_id in range(1, 8):
            scraped_at = f"2026-02-2{user_id}T00:00:00+00:00"
            user_anime_rows.extend(
                [
                    (user_id, 1, 9, 2, 26, scraped_at),
                    (user_id, 2, 8, 2, 26, scraped_at),
                    (user_id, 3, 8, 2, 22, scraped_at),
                    (user_id, 4, 7, 2, 13, scraped_at),
                    (user_id, 5, 6, 1, 8, scraped_at),
                ]
            )
        conn.executemany(
            """
            INSERT INTO user_anime(
                user_id, anime_id, score, watching_status, watched_episodes, scraped_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            user_anime_rows,
        )
        conn.commit()


def _make_semantic_ids(path: Path) -> None:
    rows = [
        {
            "anime_id": 1,
            "name": "Cowboy Bebop",
            "english_name": "Cowboy Bebop",
            "genres": "Action, Sci-Fi, Drama",
            "semantic_id": "<L1_1><L2_1><L3_1><L4_1><L5_1><L6_1><L7_1><L8_1>",
            "tokens": [f"<L{i}_1>" for i in range(1, 9)],
            "codes": [1] * 8,
        },
        {
            "anime_id": 2,
            "name": "Neon Genesis Evangelion",
            "english_name": "Neon Genesis Evangelion",
            "genres": "Drama, Sci-Fi, Psychological",
            "semantic_id": "<L1_2><L2_2><L3_2><L4_2><L5_2><L6_2><L7_2><L8_2>",
            "tokens": [f"<L{i}_2>" for i in range(1, 9)],
            "codes": [2] * 8,
        },
        {
            "anime_id": 3,
            "name": "Psycho-Pass",
            "english_name": "Psycho-Pass",
            "genres": "Sci-Fi, Psychological, Thriller",
            "semantic_id": "<L1_3><L2_3><L3_3><L4_3><L5_3><L6_3><L7_3><L8_3>",
            "tokens": [f"<L{i}_3>" for i in range(1, 9)],
            "codes": [3] * 8,
        },
        {
            "anime_id": 4,
            "name": "Serial Experiments Lain",
            "english_name": "Serial Experiments Lain",
            "genres": "Psychological, Mystery, Sci-Fi",
            "semantic_id": "<L1_4><L2_4><L3_4><L4_4><L5_4><L6_4><L7_4><L8_4>",
            "tokens": [f"<L{i}_4>" for i in range(1, 9)],
            "codes": [4] * 8,
        },
        {
            "anime_id": 5,
            "name": "Trigun",
            "english_name": "Trigun",
            "genres": "Action, Adventure, Sci-Fi",
            "semantic_id": "<L1_5><L2_5><L3_5><L4_5><L5_5><L6_5><L7_5><L8_5>",
            "tokens": [f"<L{i}_5>" for i in range(1, 9)],
            "codes": [5] * 8,
        },
        {
            "anime_id": 6,
            "name": "Samurai Champloo",
            "english_name": "Samurai Champloo",
            "genres": "Action, Adventure",
            "semantic_id": "<L1_6><L2_6><L3_6><L4_6><L5_6><L6_6><L7_6><L8_6>",
            "tokens": [f"<L{i}_6>" for i in range(1, 9)],
            "codes": [6] * 8,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_prep_builds_expected_task_split(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    semantic_ids = tmp_path / "semantic_ids.jsonl"
    out_dir = tmp_path / "out"
    _make_source_db(source_db)
    _make_semantic_ids(semantic_ids)

    run_prep(
        PrepConfig(
            source_db=source_db,
            semantic_ids_path=semantic_ids,
            out_dir=out_dir,
            rebuild=True,
            seed=123,
            target_examples=40,
            split_task_a=0.50,
            split_task_b=0.25,
            split_task_c=0.25,
            task_b_min_history=4,
            task_b_max_history=10,
            task_b_mask_modes=["last", "random"],
            export_parquet=False,
            write_manifest=True,
        )
    )

    output_path = out_dir / "llm_prep_train.jsonl"
    summary_path = out_dir / "llm_prep_summary.json"
    manifest_path = out_dir / "llm_prep_manifest.json"
    assert output_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()

    with output_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 40

    task_counts = Counter(row["task"] for row in rows)
    assert task_counts["A"] == 20
    assert task_counts["B"] == 10
    assert task_counts["C"] == 10
    assert any("[MASK]" in row["input"] for row in rows if row["task"] == "B")
    assert all(row["output"] for row in rows)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["task_a"] == 20
    assert summary["counts"]["task_b"] == 10
    assert summary["counts"]["task_c"] == 10


def test_prep_reuse_mode_keeps_existing_output(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    semantic_ids = tmp_path / "semantic_ids.jsonl"
    out_dir = tmp_path / "out"
    _make_source_db(source_db)
    _make_semantic_ids(semantic_ids)

    config = PrepConfig(
        source_db=source_db,
        semantic_ids_path=semantic_ids,
        out_dir=out_dir,
        rebuild=True,
        seed=7,
        target_examples=24,
        split_task_a=0.50,
        split_task_b=0.25,
        split_task_c=0.25,
        task_b_min_history=4,
        task_b_max_history=10,
        task_b_mask_modes=["last", "random"],
    )
    run_prep(config)

    output_path = out_dir / "llm_prep_train.jsonl"
    before = hashlib.sha256(output_path.read_bytes()).hexdigest()

    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO user_anime(user_id, anime_id, score, watching_status, watched_episodes, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (999, 6, 10, 2, 12, "2026-02-26T00:00:00+00:00"),
        )
        conn.commit()

    run_prep(
        PrepConfig(
            source_db=source_db,
            semantic_ids_path=semantic_ids,
            out_dir=out_dir,
            rebuild=False,
            seed=7,
            target_examples=24,
            split_task_a=0.50,
            split_task_b=0.25,
            split_task_c=0.25,
            task_b_min_history=4,
            task_b_max_history=10,
            task_b_mask_modes=["last", "random"],
        )
    )

    after = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert before == after
