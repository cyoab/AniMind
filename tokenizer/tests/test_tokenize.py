from __future__ import annotations

import json
import re
import sqlite3
from array import array
from pathlib import Path

from typer.testing import CliRunner

import animind_tokenizer.cli as cli_module
from animind_tokenizer.cli import app
from animind_tokenizer.rqvae import RQVAEConfig, run_rqvae
from animind_tokenizer.tokenize import TokenizeConfig, run_tokenize


def _vector_to_blob(vector: list[float]) -> bytes:
    packed = array("f", [float(v) for v in vector])
    return packed.tobytes()


def _create_tokenizer_db(path: Path, *, rows: int, dim: int) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE anime_prep (
                anime_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                english_name TEXT,
                genres TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE anime_embeddings (
                anime_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                embedded_at TEXT NOT NULL,
                source_prepared_at TEXT
            )
            """
        )
        prep_values = []
        embed_values = []
        for anime_id in range(1, rows + 1):
            prep_values.append(
                (
                    anime_id,
                    f"Anime {anime_id}",
                    f"Anime EN {anime_id}",
                    "Action, Comedy" if anime_id % 2 == 0 else "Drama, Fantasy",
                )
            )
            vector = [float(anime_id + offset) / float(dim) for offset in range(dim)]
            embed_values.append(
                (
                    anime_id,
                    sqlite3.Binary(_vector_to_blob(vector)),
                    dim,
                    "dummy/model",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                )
            )
        conn.executemany(
            "INSERT INTO anime_prep(anime_id, name, english_name, genres) VALUES (?, ?, ?, ?)",
            prep_values,
        )
        conn.executemany(
            """
            INSERT INTO anime_embeddings(
                anime_id, embedding, embedding_dim, model_name, embedded_at, source_prepared_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            embed_values,
        )
        conn.commit()


def _create_source_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE user_anime (
                user_id INTEGER NOT NULL,
                anime_id INTEGER NOT NULL,
                score INTEGER NOT NULL,
                watching_status INTEGER NOT NULL,
                watched_episodes INTEGER NOT NULL
            )
            """
        )
        rows = [
            (1, 1, 9, 2, 12),
            (1, 2, 8, 2, 12),
            (1, 3, 8, 2, 12),
            (2, 1, 8, 2, 24),
            (2, 2, 7, 2, 24),
            (2, 4, 8, 2, 24),
            (3, 1, 9, 2, 10),
            (3, 3, 8, 2, 10),
            (3, 5, 8, 2, 10),
            (4, 6, 9, 2, 26),
            (4, 7, 8, 2, 26),
            (4, 8, 8, 2, 26),
            (5, 9, 9, 2, 12),
            (5, 10, 8, 2, 12),
            (5, 11, 8, 2, 12),
            (6, 12, 6, 2, 12),  # filtered by score_min
            (6, 2, 9, 1, 2),  # filtered by status
        ]
        conn.executemany(
            """
            INSERT INTO user_anime(user_id, anime_id, score, watching_status, watched_episodes)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _train_checkpoint(tokenizer_db: Path, checkpoint_dir: Path, *, dim: int) -> Path:
    run_rqvae(
        RQVAEConfig(
            tokenizer_db=tokenizer_db,
            out_dir=checkpoint_dir,
            rebuild=True,
            device="cpu",
            batch_size=4,
            epochs=1,
            num_workers=0,
            val_ratio=0.2,
            lr=0.00004,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
        )
    )
    checkpoint = checkpoint_dir / "rqvae_best.pt"
    assert checkpoint.exists()
    return checkpoint


def test_run_tokenize_creates_expected_outputs(tmp_path: Path) -> None:
    tokenizer_db = tmp_path / "tokenizer.sqlite"
    source_db = tmp_path / "anilist.sqlite"
    checkpoint_dir = tmp_path / "rqvae"
    out_dir = tmp_path / "tokenize"

    _create_tokenizer_db(tokenizer_db, rows=12, dim=6)
    _create_source_db(source_db)
    checkpoint = _train_checkpoint(tokenizer_db, checkpoint_dir, dim=6)

    run_tokenize(
        TokenizeConfig(
            tokenizer_db=tokenizer_db,
            source_db=source_db,
            rqvae_checkpoint=checkpoint,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            batch_size=4,
            write_db_tables=True,
            cluster_sample_size=3,
            cluster_min_bucket=1,
            recall_k=3,
            recall_min_support=1,
            recall_max_queries=20,
        )
    )

    required = [
        out_dir / "semantic_ids.jsonl",
        out_dir / "semantic_lookup.tsv",
        out_dir / "semantic_id_to_anime.jsonl",
        out_dir / "semantic_vocab.json",
        out_dir / "special_tokens.json",
        out_dir / "token_stats.json",
        out_dir / "cluster_inspection.jsonl",
        out_dir / "cluster_summary.json",
        out_dir / "recall_report.json",
        out_dir / "tokenize_config.json",
        out_dir / "tokenize_run_summary.json",
    ]
    for path in required:
        assert path.exists(), f"missing artifact: {path}"

    semantic_lines = (
        (out_dir / "semantic_ids.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(semantic_lines) == 12
    first = json.loads(semantic_lines[0])
    assert re.match(r"^<L1_\d+><L2_\d+>$", first["semantic_id"]) is not None
    assert first["semantic_id"] == "".join(first["tokens"])
    assert len(first["codes"]) == 2

    vocab = json.loads((out_dir / "semantic_vocab.json").read_text(encoding="utf-8"))
    assert "<anime_start>" in vocab["tokens"]
    assert "<watch>" in vocab["tokens"]
    assert "<L1_0>" in vocab["tokens"]
    assert "<L2_0>" in vocab["tokens"]

    with sqlite3.connect(tokenizer_db) as conn:
        table_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='anime_semantic_ids'"
        ).fetchone()[0]
        assert table_count == 1
        row_count = conn.execute("SELECT COUNT(*) FROM anime_semantic_ids").fetchone()[0]
        assert row_count == 12

    recall_report = json.loads((out_dir / "recall_report.json").read_text(encoding="utf-8"))
    assert recall_report["status"] in {"ok", "skipped"}


def test_run_tokenize_dry_run_writes_subdir(tmp_path: Path) -> None:
    tokenizer_db = tmp_path / "tokenizer.sqlite"
    source_db = tmp_path / "anilist.sqlite"
    checkpoint_dir = tmp_path / "rqvae"
    out_dir = tmp_path / "tokenize"

    _create_tokenizer_db(tokenizer_db, rows=10, dim=6)
    _create_source_db(source_db)
    checkpoint = _train_checkpoint(tokenizer_db, checkpoint_dir, dim=6)

    run_tokenize(
        TokenizeConfig(
            tokenizer_db=tokenizer_db,
            source_db=source_db,
            rqvae_checkpoint=checkpoint,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            batch_size=4,
            cluster_min_bucket=1,
            recall_min_support=1,
            dry_run=True,
            dry_run_limit=6,
            dry_run_out_subdir="quickcheck",
        )
    )

    dry_dir = out_dir / "quickcheck"
    assert (dry_dir / "semantic_ids.jsonl").exists()
    assert (dry_dir / "semantic_lookup.tsv").exists()
    assert (dry_dir / "semantic_vocab.json").exists()
    assert (dry_dir / "recall_report.json").exists()


def test_cli_tokenize_phase_smoke(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "tokenize"
    source_db = tmp_path / "anilist.sqlite"
    checkpoint = tmp_path / "rqvae_best.pt"

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE anime_prep (anime_id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE anime_embeddings ("
            "anime_id INTEGER PRIMARY KEY, "
            "embedding BLOB, "
            "embedding_dim INTEGER)"
        )
        conn.commit()
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            "CREATE TABLE user_anime ("
            "user_id INTEGER, anime_id INTEGER, score INTEGER, watching_status INTEGER)"
        )
        conn.commit()
    checkpoint.write_bytes(b"stub")

    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                f'source_db = "{source_db}"',
                f'out_dir = "{tmp_path}"',
                "rebuild = true",
                "export_parquet = false",
                "",
                "[embedd]",
                f'tokenizer_db = "{db_path}"',
                "rebuild = true",
                "",
                "[rqvae]",
                f'tokenizer_db = "{db_path}"',
                f'out_dir = "{tmp_path / "rqvae"}"',
                "rebuild = true",
                "batch_size = 2",
                "epochs = 1",
                "num_workers = 0",
                "val_ratio = 0.2",
                "warmup_steps = 0",
                "latent_dim = 8",
                "rq_levels = 2",
                "codebook_size = 32",
                "encoder_hidden_dim = 16",
                "decoder_hidden_dim = 16",
                "",
                "[tokenize]",
                f'tokenizer_db = "{db_path}"',
                f'source_db = "{source_db}"',
                f'rqvae_checkpoint = "{checkpoint}"',
                f'out_dir = "{out_dir}"',
                "rebuild = true",
                "limit = 0",
                'device = "cpu"',
                "batch_size = 4",
                "recall_k = 5",
                "recall_min_support = 1",
            ]
        ),
        encoding="utf-8",
    )

    called = {"tokenize": False}

    def _patched_run_tokenize(config: object) -> None:
        called["tokenize"] = True

    monkeypatch.setattr(cli_module, "run_tokenize", _patched_run_tokenize)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--phase",
            "tokenize",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert called["tokenize"] is True
