from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from animind_tokenizer.cli import app
from animind_tokenizer.prep import PrepConfig, run_prep

ANIME_SCHEMA_SQL = """
CREATE TABLE anime (
    anime_id INTEGER PRIMARY KEY,
    name TEXT,
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
    imported_at TEXT NOT NULL
)
"""


def _make_source_db(path: Path, rows: list[tuple]) -> None:
    expanded_rows: list[tuple] = []
    for row in rows:
        if len(row) == 16:
            (
                anime_id,
                name,
                english_name,
                score,
                genres,
                synopsis,
                anime_type,
                episodes,
                premiered,
                studios,
                source,
                rating,
                members,
                favorites,
                scored_by,
                imported_at,
            ) = row
            expanded_rows.append(
                (
                    anime_id,
                    name,
                    english_name,
                    None,
                    None,
                    score,
                    genres,
                    synopsis,
                    anime_type,
                    episodes,
                    premiered,
                    None,
                    None,
                    studios,
                    None,
                    None,
                    source,
                    None,
                    rating,
                    None,
                    None,
                    None,
                    members,
                    favorites,
                    scored_by,
                    None,
                    None,
                    imported_at,
                )
            )
        elif len(row) == 28:
            expanded_rows.append(row)
        else:
            raise ValueError(f"Unexpected row length: {len(row)}")

    with sqlite3.connect(path) as conn:
        conn.execute(ANIME_SCHEMA_SQL)
        conn.executemany(
            """
            INSERT INTO anime(
                anime_id, name, english_name, japanese_name, title_synonyms, score, genres, synopsis, type, episodes,
                premiered, season, year, studios, producers, licensors, source, anime_status, rating, duration, aired_from, aired_to,
                members, favorites, scored_by, rank, popularity, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            expanded_rows,
        )
        conn.commit()


def test_prep_creates_output_table(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                1,
                "Death Note",
                "Death Note",
                8.6,
                "Mystery, Thriller",
                "A notebook of death.",
                "TV",
                37,
                "Fall 2006",
                "Madhouse",
                "Manga",
                "R - 17+",
                1500000,
                80000,
                900000,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )
    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='anime_prep'"
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) FROM anime_prep").fetchone()[0]

    assert exists is not None
    assert count == 1


def test_cleaning_normalizes_text_and_numbers(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                2,
                "  Name;With ; Delimiters  ",
                " \tEnglish\nName ",
                11.5,
                "Action;\nComedy",
                "  Some\tplot\ntext  ",
                " TV ",
                -1,
                " nan ",
                " Studio;A ",
                " none ",
                "null",
                "not-a-number",
                -7,
                20.8,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        row = conn.execute(
            """
            SELECT
                name, english_name, score, genres, synopsis, type, episodes, premiered,
                studios, source, rating, members, favorites, scored_by
            FROM anime_prep WHERE anime_id = 2
            """
        ).fetchone()

    assert row[0] == "Name,With , Delimiters"
    assert row[1] == "English Name"
    assert row[2] is None
    assert row[3] == "Action, Comedy"
    assert row[4] == "Some plot text"
    assert row[5] == "TV"
    assert row[6] is None
    assert row[7] is None
    assert row[8] == "Studio,A"
    assert row[9] is None
    assert row[10] is None
    assert row[11] is None
    assert row[12] is None
    assert row[13] == 20


def test_flags_are_set_correctly(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                3,
                None,
                None,
                6.5,
                "Drama",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        row = conn.execute(
            "SELECT name, flag_missing_core, flag_missing_synopsis FROM anime_prep WHERE anime_id = 3"
        ).fetchone()

    assert row[0] == "Unknown"
    assert row[1] == 1
    assert row[2] == 1


def test_anime_text_is_deterministic(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                4,
                "Cowboy Bebop",
                "Cowboy Bebop",
                8.75,
                "Action, Sci-Fi",
                "Space bounty hunters.",
                "TV",
                26,
                "Spring 1998",
                "Sunrise",
                "Original",
                "R - 17+",
                900000,
                65000,
                500000,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        text = conn.execute("SELECT anime_text FROM anime_prep WHERE anime_id = 4").fetchone()[0]

    expected = (
        "Anime ID: 4; Name: Cowboy Bebop; English Name: Cowboy Bebop; "
        "Japanese Name: Unknown; Title Synonyms: Unknown; Score: 8.75; "
        "Genres: Action, Sci-Fi; Synopsis: Space bounty hunters.; Type: TV; Episodes: 26; "
        "Premiered: Spring 1998; Season: Unknown; Year: Unknown; Studios: Sunrise; Producers: Unknown; "
        "Licensors: Unknown; Source: Original; Status: Unknown; Rating: R - 17+; Duration: Unknown; "
        "Aired From: Unknown; Aired To: Unknown; Members: 900000; Favorites: 65000; Scored By: 500000; "
        "Rank: Unknown; Popularity: Unknown."
    )
    assert text == expected


def test_unknown_placeholders_are_treated_as_missing(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                7,
                "Placeholder Anime",
                "English",
                "Japanese",
                "Syn1, Syn2",
                6.8,
                "Action",
                "Plot",
                "Unknown",
                12,
                "Fall 2021",
                "Fall",
                2021,
                "Studio",
                "Prod",
                "Lic",
                "Manga",
                "Unknown",
                "Unknown",
                "24 min per ep",
                "2021-10-01",
                "2021-12-01",
                12345,
                200,
                1000,
                500,
                2500,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        row = conn.execute(
            "SELECT type, anime_status, rating, flag_missing_core FROM anime_prep WHERE anime_id = 7"
        ).fetchone()

    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] == 1


def test_reuse_mode_skips_rebuild(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                5,
                "Initial",
                None,
                7.0,
                None,
                None,
                "TV",
                12,
                None,
                None,
                None,
                "PG-13",
                100,
                10,
                5,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=True, export_parquet=False))

    with sqlite3.connect(source_db) as conn:
        conn.execute("UPDATE anime SET name = 'Updated' WHERE anime_id = 5")
        conn.commit()

    run_prep(PrepConfig(source_db=source_db, out_dir=out_dir, rebuild=False, export_parquet=False))

    with sqlite3.connect(out_dir / "tokenizer.sqlite") as conn:
        name = conn.execute("SELECT name FROM anime_prep WHERE anime_id = 5").fetchone()[0]
        count = conn.execute("SELECT COUNT(*) FROM anime_prep").fetchone()[0]

    assert name == "Initial"
    assert count == 1


def test_cli_run_phase_prep_smoke(tmp_path: Path) -> None:
    source_db = tmp_path / "anilist.sqlite"
    out_dir = tmp_path / "tokenizer_output"
    _make_source_db(
        source_db,
        [
            (
                6,
                "CLI Anime",
                None,
                7.5,
                "Adventure",
                "Text",
                "TV",
                24,
                "Winter 2020",
                "Studio",
                "Original",
                "PG-13",
                1000,
                50,
                300,
                "2026-02-25T00:00:00+00:00",
            )
        ],
    )

    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                f'source_db = "{source_db}"',
                f'out_dir = "{out_dir}"',
                "rebuild = true",
                "export_parquet = true",
                "limit = 0",
                "",
                "[embedd]",
                f'tokenizer_db = "{out_dir / "tokenizer.sqlite"}"',
                "rebuild = true",
                "limit = 0",
                'model_name = "dummy/model"',
                "batch_size = 2",
                "max_length = 128",
                'device = "cpu"',
                "normalize = true",
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
    assert (out_dir / "tokenizer.sqlite").exists()
    assert (out_dir / "anime_prep.parquet").exists()
