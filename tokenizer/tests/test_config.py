from __future__ import annotations

from pathlib import Path

from animind_tokenizer.config import (
    build_embed_config,
    build_prep_config,
    build_rqvae_config,
    build_tokenize_config,
)


def test_build_configs_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'source_db = "../output/anilist.sqlite"',
                'out_dir = "./output"',
                "rebuild = false",
                "export_parquet = false",
                "limit = 123",
                "",
                "[embedd]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
                'env_file = "./.env"',
                "rebuild = true",
                "limit = 321",
                'model_name = "test/model"',
                "batch_size = 16",
                "max_length = 1024",
                'device = "cuda"',
                'precision = "bfloat16"',
                'hf_token = ""',
                'hf_token_env = "HF_TOKEN"',
                "normalize = false",
                "",
                "[rqvae]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
                'out_dir = "./output/rqvae"',
                "rebuild = true",
                "limit = 100",
                'device = "cuda:0"',
                "seed = 123",
                "batch_size = 128",
                "epochs = 10",
                "num_workers = 2",
                "val_ratio = 0.1",
                "lr = 0.00004",
                "adam_beta1 = 0.5",
                "adam_beta2 = 0.9",
                "warmup_steps = 100",
                "latent_dim = 256",
                "rq_levels = 8",
                "codebook_size = 2048",
                "commitment_beta = 0.25",
                "ema_decay = 0.99",
                "ema_eps = 0.00001",
                "restart_unused_codes = true",
                "amp = true",
                "checkpoint_every = 1",
                "encoder_hidden_dim = 1024",
                "decoder_hidden_dim = 1024",
                "dry_run = true",
                "dry_run_limit = 64",
                "dry_run_epochs = 2",
                "dry_run_batch_size = 8",
                "dry_run_num_workers = 0",
                'dry_run_out_subdir = "quickcheck"',
                'env_file = "./.env"',
                "wandb_enabled = true",
                'wandb_mode = "offline"',
                'wandb_project = "animind-tokenizer"',
                'wandb_entity = "team-alpha"',
                'wandb_api_key = ""',
                'wandb_run_name = "rqvae-test"',
                'wandb_group = "ci"',
                'wandb_tags = ["rqvae", "offline"]',
                'wandb_project_env = "WANDB_PROJECT"',
                'wandb_entity_env = "WANDB_ENTITY"',
                'wandb_api_key_env = "WANDB_API_KEY"',
                'resume_from = "last"',
                "resume_strict = true",
                "",
                "[tokenize]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
                'source_db = "../output/anilist.sqlite"',
                'rqvae_checkpoint = "./output/rqvae/rqvae_best.pt"',
                'out_dir = "./output/tokenize"',
                "rebuild = true",
                "limit = 11",
                'device = "cuda"',
                "batch_size = 64",
                "write_db_tables = true",
                'special_tokens = ["<anime_start>", "<anime_end>"]',
                "semantic_id_concat = true",
                'semantic_id_separator = ""',
                "cluster_sample_size = 5",
                "cluster_min_bucket = 8",
                "cluster_random_seed = 777",
                "recall_k = 30",
                "recall_min_support = 4",
                "recall_positive_score_min = 8",
                "recall_completed_status = 2",
                "recall_max_queries = 100",
                "recall_max_rows = 12345",
                "recall_seed = 888",
                "dry_run = true",
                "dry_run_limit = 32",
                'dry_run_out_subdir = "quickcheck"',
            ]
        ),
        encoding="utf-8",
    )

    prep_cfg = build_prep_config(config_path=config_path)
    embed_cfg = build_embed_config(config_path=config_path)
    rqvae_cfg = build_rqvae_config(config_path=config_path)
    tokenize_cfg = build_tokenize_config(config_path=config_path)

    assert prep_cfg.source_db == (tmp_path / "../output/anilist.sqlite").resolve()
    assert prep_cfg.out_dir == (tmp_path / "./output").resolve()
    assert prep_cfg.rebuild is False
    assert prep_cfg.export_parquet is False
    assert prep_cfg.limit == 123

    assert embed_cfg.tokenizer_db == (tmp_path / "./output/tokenizer.sqlite").resolve()
    assert embed_cfg.env_file == (tmp_path / "./.env").resolve()
    assert embed_cfg.rebuild is True
    assert embed_cfg.limit == 321
    assert embed_cfg.model_name == "test/model"
    assert embed_cfg.batch_size == 16
    assert embed_cfg.max_length == 1024
    assert embed_cfg.device == "cuda"
    assert embed_cfg.precision == "bfloat16"
    assert embed_cfg.hf_token == ""
    assert embed_cfg.hf_token_env == "HF_TOKEN"
    assert embed_cfg.normalize is False

    assert rqvae_cfg.tokenizer_db == (tmp_path / "./output/tokenizer.sqlite").resolve()
    assert rqvae_cfg.out_dir == (tmp_path / "./output/rqvae").resolve()
    assert rqvae_cfg.device == "cuda:0"
    assert rqvae_cfg.batch_size == 128
    assert rqvae_cfg.epochs == 10
    assert rqvae_cfg.rq_levels == 8
    assert rqvae_cfg.codebook_size == 2048
    assert rqvae_cfg.env_file == (tmp_path / "./.env").resolve()
    assert rqvae_cfg.dry_run is True
    assert rqvae_cfg.dry_run_limit == 64
    assert rqvae_cfg.dry_run_out_subdir == "quickcheck"
    assert rqvae_cfg.wandb_enabled is True
    assert rqvae_cfg.wandb_mode == "offline"
    assert rqvae_cfg.wandb_project == "animind-tokenizer"
    assert rqvae_cfg.wandb_entity == "team-alpha"
    assert rqvae_cfg.wandb_tags == ["rqvae", "offline"]
    assert rqvae_cfg.resume_from == "last"
    assert rqvae_cfg.resume_strict is True

    assert tokenize_cfg.tokenizer_db == (tmp_path / "./output/tokenizer.sqlite").resolve()
    assert tokenize_cfg.source_db == (tmp_path / "../output/anilist.sqlite").resolve()
    assert tokenize_cfg.rqvae_checkpoint == (tmp_path / "./output/rqvae/rqvae_best.pt").resolve()
    assert tokenize_cfg.out_dir == (tmp_path / "./output/tokenize").resolve()
    assert tokenize_cfg.limit == 11
    assert tokenize_cfg.device == "cuda"
    assert tokenize_cfg.batch_size == 64
    assert tokenize_cfg.write_db_tables is True
    assert tokenize_cfg.special_tokens == ["<anime_start>", "<anime_end>"]
    assert tokenize_cfg.cluster_sample_size == 5
    assert tokenize_cfg.cluster_min_bucket == 8
    assert tokenize_cfg.recall_k == 30
    assert tokenize_cfg.recall_positive_score_min == 8
    assert tokenize_cfg.recall_max_queries == 100
    assert tokenize_cfg.recall_max_rows == 12345
    assert tokenize_cfg.dry_run is True
    assert tokenize_cfg.dry_run_limit == 32
    assert tokenize_cfg.dry_run_out_subdir == "quickcheck"


def test_missing_embedd_section_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'source_db = "../output/anilist.sqlite"',
                'out_dir = "./output"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        build_embed_config(config_path=config_path)
    except RuntimeError as exc:
        assert "[embedd]" in str(exc)
    else:
        raise AssertionError("Expected missing [embedd] section to raise RuntimeError.")


def test_missing_rqvae_section_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'source_db = "../output/anilist.sqlite"',
                'out_dir = "./output"',
                "",
                "[embedd]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        build_rqvae_config(config_path=config_path)
    except RuntimeError as exc:
        assert "[rqvae]" in str(exc)
    else:
        raise AssertionError("Expected missing [rqvae] section to raise RuntimeError.")


def test_missing_tokenize_section_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'source_db = "../output/anilist.sqlite"',
                'out_dir = "./output"',
                "",
                "[embedd]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
                "",
                "[rqvae]",
                'tokenizer_db = "./output/tokenizer.sqlite"',
                'out_dir = "./output/rqvae"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        build_tokenize_config(config_path=config_path)
    except RuntimeError as exc:
        assert "[tokenize]" in str(exc)
    else:
        raise AssertionError("Expected missing [tokenize] section to raise RuntimeError.")
