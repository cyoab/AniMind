from __future__ import annotations

import sqlite3
from array import array
import math
from pathlib import Path

import torch
from typer.testing import CliRunner

import animind_tokenizer.cli as cli_module
from animind_tokenizer.cli import app
from animind_tokenizer.rqvae import (
    RQVAEConfig,
    _RQVAEModel,
    _should_update_best,
    _step_optimizer_and_scheduler,
    load_rqvae_for_eval,
    run_rqvae,
)


def _vector_to_blob(vector: list[float]) -> bytes:
    packed = array("f", [float(v) for v in vector])
    return packed.tobytes()


def _create_embedding_db(path: Path, *, rows: int, dim: int) -> None:
    with sqlite3.connect(path) as conn:
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
        values = []
        for anime_id in range(1, rows + 1):
            vector = [float(anime_id + offset) / float(dim) for offset in range(dim)]
            values.append(
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
            """
            INSERT INTO anime_embeddings(
                anime_id, embedding, embedding_dim, model_name, embedded_at, source_prepared_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        conn.commit()


def test_rqvae_model_forward_and_gradients() -> None:
    config = RQVAEConfig(
        latent_dim=4,
        rq_levels=3,
        codebook_size=32,
        commitment_beta=0.25,
        encoder_hidden_dim=12,
        decoder_hidden_dim=12,
    )
    model = _RQVAEModel(input_dim=6, config=config)
    model.train()
    batch = torch.randn(5, 6)
    outputs = model(batch)
    outputs["loss"].backward()

    assert outputs["codes"].shape == (5, 3)
    assert outputs["usage_counts"].shape == (32,)
    assert model.quantizer.codebook.shape == (32, 4)
    first_weight = model.encoder.net[0].weight
    assert first_weight.grad is not None
    assert torch.count_nonzero(first_weight.grad).item() > 0


def test_run_rqvae_creates_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    _create_embedding_db(db_path, rows=24, dim=6)

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            seed=7,
            batch_size=8,
            epochs=2,
            num_workers=0,
            val_ratio=0.25,
            lr=0.00004,
            warmup_steps=0,
            latent_dim=4,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
        )
    )

    assert (out_dir / "rqvae_best.pt").exists()
    assert (out_dir / "rqvae_last.pt").exists()
    assert (out_dir / "rqvae_config.json").exists()
    assert (out_dir / "rqvae_metrics.jsonl").exists()
    assert (out_dir / "code_usage.csv").exists()

    metrics_lines = (out_dir / "rqvae_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2


def test_run_rqvae_saves_last_at_training_end_even_if_interval_skips(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    _create_embedding_db(db_path, rows=24, dim=6)

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            batch_size=8,
            epochs=3,
            num_workers=0,
            val_ratio=0.25,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=2,
        )
    )

    payload = torch.load(out_dir / "rqvae_last.pt", map_location="cpu")
    assert int(payload["epoch"]) == 3


def test_run_rqvae_dry_run_creates_subdir_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    _create_embedding_db(db_path, rows=48, dim=6)

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            seed=7,
            batch_size=16,
            epochs=4,
            num_workers=0,
            val_ratio=0.25,
            lr=0.00004,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
            dry_run=True,
            dry_run_limit=12,
            dry_run_epochs=1,
            dry_run_batch_size=4,
            dry_run_num_workers=0,
            dry_run_out_subdir="quickcheck",
        )
    )

    dry_dir = out_dir / "quickcheck"
    assert (dry_dir / "rqvae_best.pt").exists()
    assert (dry_dir / "rqvae_last.pt").exists()
    assert (dry_dir / "rqvae_metrics.jsonl").exists()
    metrics_lines = (dry_dir / "rqvae_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 1


def test_run_rqvae_wandb_enabled_without_project_fails(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    _create_embedding_db(db_path, rows=12, dim=6)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    try:
        run_rqvae(
            RQVAEConfig(
                tokenizer_db=db_path,
                out_dir=tmp_path / "rqvae",
                rebuild=True,
                device="cpu",
                batch_size=4,
                epochs=1,
                num_workers=0,
                val_ratio=0.25,
                warmup_steps=0,
                latent_dim=8,
                rq_levels=2,
                codebook_size=32,
                encoder_hidden_dim=16,
                decoder_hidden_dim=16,
                wandb_enabled=True,
                wandb_mode="offline",
                wandb_project="",
                env_file=tmp_path / "missing.env",
            )
        )
    except RuntimeError as exc:
        assert "project is missing" in str(exc)
    else:
        raise AssertionError("Expected run_rqvae to fail when wandb is enabled but project is unset.")


def test_should_update_best_handles_non_finite_values() -> None:
    assert _should_update_best(epoch=1, current_val=float("nan"), best_val=math.inf) is True
    assert _should_update_best(epoch=2, current_val=float("nan"), best_val=1.0) is False
    assert _should_update_best(epoch=2, current_val=0.5, best_val=float("nan")) is True
    assert _should_update_best(epoch=2, current_val=1.5, best_val=1.0) is False


def test_step_optimizer_and_scheduler_skips_scheduler_when_scale_drops() -> None:
    class FakeOptimizer:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    class FakeScheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    class FakeScaler:
        def __init__(self, before: float, after: float) -> None:
            self.scale = before
            self.after = after

        def get_scale(self) -> float:
            return self.scale

        def step(self, optimizer: FakeOptimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            self.scale = self.after

    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    scaler = FakeScaler(before=8.0, after=4.0)
    stepped = _step_optimizer_and_scheduler(
        optimizer=optimizer,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        scaler=scaler,  # type: ignore[arg-type]
    )

    assert stepped is False
    assert optimizer.steps == 1
    assert scheduler.steps == 0


def test_step_optimizer_and_scheduler_steps_scheduler_when_scale_holds() -> None:
    class FakeOptimizer:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    class FakeScheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    class FakeScaler:
        def __init__(self, before: float, after: float) -> None:
            self.scale = before
            self.after = after

        def get_scale(self) -> float:
            return self.scale

        def step(self, optimizer: FakeOptimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            self.scale = self.after

    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    scaler = FakeScaler(before=8.0, after=8.0)
    stepped = _step_optimizer_and_scheduler(
        optimizer=optimizer,  # type: ignore[arg-type]
        scheduler=scheduler,  # type: ignore[arg-type]
        scaler=scaler,  # type: ignore[arg-type]
    )

    assert stepped is True
    assert optimizer.steps == 1
    assert scheduler.steps == 1


def test_run_rqvae_resume_from_last_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    _create_embedding_db(db_path, rows=24, dim=6)

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            batch_size=8,
            epochs=1,
            num_workers=0,
            val_ratio=0.25,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
        )
    )

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=False,
            resume_from="last",
            device="cpu",
            batch_size=8,
            epochs=2,
            num_workers=0,
            val_ratio=0.25,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
        )
    )

    payload = torch.load(out_dir / "rqvae_last.pt", map_location="cpu")
    assert int(payload["epoch"]) == 2
    metrics_lines = (out_dir / "rqvae_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2


def test_load_rqvae_for_eval_from_best_checkpoint(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    _create_embedding_db(db_path, rows=24, dim=6)

    run_rqvae(
        RQVAEConfig(
            tokenizer_db=db_path,
            out_dir=out_dir,
            rebuild=True,
            device="cpu",
            batch_size=8,
            epochs=1,
            num_workers=0,
            val_ratio=0.25,
            warmup_steps=0,
            latent_dim=8,
            rq_levels=2,
            codebook_size=32,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            checkpoint_every=1,
        )
    )

    model, checkpoint = load_rqvae_for_eval(
        checkpoint_path=out_dir / "rqvae_best.pt",
        device="cpu",
    )
    assert model.training is False
    assert int(checkpoint["epoch"]) == 1
    with torch.no_grad():
        out = model(torch.randn(3, 6))
    assert "loss" in out
    assert out["codes"].shape == (3, 2)


def test_run_rqvae_rejects_non_finite_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE anime_embeddings (
                anime_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO anime_embeddings(anime_id, embedding, embedding_dim) VALUES (?, ?, ?)",
            (1, sqlite3.Binary(_vector_to_blob([math.nan, 0.1, 0.2])), 3),
        )
        conn.commit()

    try:
        run_rqvae(
            RQVAEConfig(
                tokenizer_db=db_path,
                out_dir=tmp_path / "rqvae",
                rebuild=True,
                device="cpu",
                batch_size=2,
                epochs=1,
                num_workers=0,
                latent_dim=8,
                rq_levels=2,
                codebook_size=32,
                encoder_hidden_dim=8,
                decoder_hidden_dim=8,
            )
        )
    except RuntimeError as exc:
        assert "contains non-finite values" in str(exc)
    else:
        raise AssertionError("Expected run_rqvae to fail on non-finite embeddings.")


def test_run_rqvae_rejects_inconsistent_embedding_dim(tmp_path: Path) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE anime_embeddings (
                anime_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO anime_embeddings(anime_id, embedding, embedding_dim) VALUES (?, ?, ?)",
            (1, sqlite3.Binary(_vector_to_blob([0.1, 0.2, 0.3])), 3),
        )
        conn.execute(
            "INSERT INTO anime_embeddings(anime_id, embedding, embedding_dim) VALUES (?, ?, ?)",
            (2, sqlite3.Binary(_vector_to_blob([0.1, 0.2])), 2),
        )
        conn.commit()

    try:
        run_rqvae(
            RQVAEConfig(
                tokenizer_db=db_path,
                out_dir=tmp_path / "rqvae",
                rebuild=True,
                device="cpu",
                batch_size=2,
                epochs=1,
                num_workers=0,
                latent_dim=2,
                rq_levels=2,
                codebook_size=32,
                encoder_hidden_dim=8,
                decoder_hidden_dim=8,
            )
        )
    except RuntimeError as exc:
        assert "inconsistent embedding_dim" in str(exc)
    else:
        raise AssertionError("Expected run_rqvae to fail on inconsistent embedding_dim.")


def test_cli_rqvae_phase_smoke(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "tokenizer.sqlite"
    out_dir = tmp_path / "rqvae"
    source_db = tmp_path / "anilist.sqlite"
    _create_embedding_db(db_path, rows=8, dim=4)
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
                "",
                "[rqvae]",
                f'tokenizer_db = "{db_path}"',
                f'out_dir = "{out_dir}"',
                "rebuild = true",
                "limit = 0",
                'device = "cpu"',
                "seed = 7",
                "batch_size = 4",
                "epochs = 1",
                "num_workers = 0",
                "val_ratio = 0.2",
                "lr = 0.00004",
                "adam_beta1 = 0.5",
                "adam_beta2 = 0.9",
                "warmup_steps = 0",
                "latent_dim = 8",
                "rq_levels = 2",
                "codebook_size = 32",
                "commitment_beta = 0.25",
                "ema_decay = 0.99",
                "ema_eps = 0.00001",
                "restart_unused_codes = true",
                "amp = false",
                "checkpoint_every = 1",
                "encoder_hidden_dim = 16",
                "decoder_hidden_dim = 16",
            ]
        ),
        encoding="utf-8",
    )

    called = {"rqvae": False}

    def _patched_run_rqvae(config: object) -> None:
        called["rqvae"] = True

    monkeypatch.setattr(cli_module, "run_rqvae", _patched_run_rqvae)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--phase",
            "rqvae",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert called["rqvae"] is True
