from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

import animind_tokenizer.embed as embed_module
from animind_tokenizer.cli import app
from animind_tokenizer.embed import EmbedConfig, _resolve_device, blob_to_vector, run_embed


class DummyBackend:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(
                [
                    float(len(text)),
                    float(sum(ord(char) for char in text) % 1000),
                    1.5,
                ]
            )
        return vectors


class AlternateBackend:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[9.0, 9.0, 9.0] for _ in texts]


class FakeCuda:
    def __init__(self, *, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class FakeBackendsCuda:
    def __init__(self, *, built: bool = True) -> None:
        self._built = built

    def is_built(self) -> bool:
        return self._built


class FakeBackends:
    def __init__(self, *, built: bool = True) -> None:
        self.cuda = FakeBackendsCuda(built=built)


class FakeTorch:
    def __init__(self, *, available: bool, count: int, built: bool = True) -> None:
        self.cuda = FakeCuda(available=available, count=count)
        self.backends = FakeBackends(built=built)


def _create_tokenizer_db(path: Path, rows: list[tuple[int, str, str]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE anime_prep (
                anime_id INTEGER PRIMARY KEY,
                anime_text TEXT NOT NULL,
                prepared_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO anime_prep(anime_id, anime_text, prepared_at) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()


def test_embed_creates_embeddings_table(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    _create_tokenizer_db(
        db_path,
        [
            (1, "Anime ID: 1; Name: A.", "2026-01-01T00:00:00+00:00"),
            (2, "Anime ID: 2; Name: B.", "2026-01-01T00:00:00+00:00"),
        ],
    )

    run_embed(
        EmbedConfig(tokenizer_db=db_path, rebuild=True, batch_size=2, model_name="dummy/model"),
        backend=DummyBackend(),
    )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM anime_embeddings").fetchone()[0]
        row = conn.execute(
            """
            SELECT embedding, embedding_dim, model_name, source_prepared_at
            FROM anime_embeddings
            WHERE anime_id = 1
            """
        ).fetchone()

    assert count == 2
    assert row[1] == 3
    assert row[2] == "dummy/model"
    assert row[3] == "2026-01-01T00:00:00+00:00"
    vector = blob_to_vector(row[0])
    assert len(vector) == 3
    assert abs(vector[2] - 1.5) < 1e-6


def test_embed_reuse_mode_skips_existing_table(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    _create_tokenizer_db(
        db_path,
        [(1, "Anime ID: 1; Name: Keep Me.", "2026-01-01T00:00:00+00:00")],
    )

    run_embed(
        EmbedConfig(tokenizer_db=db_path, rebuild=True, model_name="dummy/model"),
        backend=DummyBackend(),
    )
    run_embed(
        EmbedConfig(tokenizer_db=db_path, rebuild=False, model_name="dummy/model"),
        backend=AlternateBackend(),
    )

    with sqlite3.connect(db_path) as conn:
        vector_blob = conn.execute(
            "SELECT embedding FROM anime_embeddings WHERE anime_id = 1"
        ).fetchone()[0]
    vector = blob_to_vector(vector_blob)
    assert vector != [9.0, 9.0, 9.0]


def test_embed_limit_is_applied(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    _create_tokenizer_db(
        db_path,
        [
            (1, "Anime ID: 1; Name: One.", "2026-01-01T00:00:00+00:00"),
            (2, "Anime ID: 2; Name: Two.", "2026-01-01T00:00:00+00:00"),
            (3, "Anime ID: 3; Name: Three.", "2026-01-01T00:00:00+00:00"),
        ],
    )

    run_embed(
        EmbedConfig(tokenizer_db=db_path, rebuild=True, limit=2, batch_size=2, model_name="dummy/model"),
        backend=DummyBackend(),
    )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM anime_embeddings").fetchone()[0]

    assert count == 2


def test_embed_requires_anime_prep_table(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    with sqlite3.connect(db_path):
        pass

    try:
        run_embed(
            EmbedConfig(tokenizer_db=db_path, rebuild=True, model_name="dummy/model"),
            backend=DummyBackend(),
        )
    except RuntimeError as exc:
        assert "anime_prep" in str(exc)
    else:
        raise AssertionError("Expected run_embed to fail when anime_prep is missing.")


def test_cli_embed_phase_smoke_with_patched_backend(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    source_db = tmp_path / "anilist.sqlite"
    _create_tokenizer_db(
        db_path,
        [
            (1, "Anime ID: 1; Name: One.", "2026-01-01T00:00:00+00:00"),
            (2, "Anime ID: 2; Name: Two.", "2026-01-01T00:00:00+00:00"),
        ],
    )
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE anime (anime_id INTEGER PRIMARY KEY)")
        conn.commit()

    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                f'source_db = "{source_db}"',
                f'out_dir = "{tmp_path}"',
                "rebuild = true",
                "export_parquet = false",
                "limit = 0",
                "",
                "[embedd]",
                f'tokenizer_db = "{db_path}"',
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

    class PatchedHFBackend:
        def __init__(self, **_: object) -> None:
            pass

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embed_module, "HuggingFaceEmbeddingBackend", PatchedHFBackend)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--phase",
            "embed",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM anime_embeddings").fetchone()[0]
    assert count == 2


def test_resolve_device_auto_prefers_cuda_when_available() -> None:
    fake_torch = FakeTorch(available=True, count=1)
    assert _resolve_device(device="auto", torch_module=fake_torch) == "cuda"


def test_resolve_device_accepts_cuda_index() -> None:
    fake_torch = FakeTorch(available=True, count=4)
    assert _resolve_device(device="cuda:2", torch_module=fake_torch) == "cuda:2"


def test_resolve_device_rejects_out_of_range_cuda_index() -> None:
    fake_torch = FakeTorch(available=True, count=1)
    try:
        _resolve_device(device="cuda:1", torch_module=fake_torch)
    except RuntimeError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("Expected _resolve_device to reject out-of-range CUDA index.")
