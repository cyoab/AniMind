from __future__ import annotations

from pathlib import Path

import pytest

from animind_recommender.config import build_prep_config, build_train_config


def test_build_prep_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'source_db = "../data/anilist.sqlite"',
                'semantic_ids_path = "../data/semantic_ids.jsonl"',
                'out_dir = "./out"',
                "rebuild = false",
                "seed = 123",
                "target_examples = 99",
                "split_task_a = 0.5",
                "split_task_b = 0.25",
                "split_task_c = 0.25",
                "max_task_a_templates_per_anime = 10",
                "task_b_min_history = 4",
                "task_b_max_history = 20",
                'task_b_mask_modes = ["last"]',
                "task_b_positive_score_min = 8",
                "task_b_allowed_statuses = [2, 4]",
                "export_parquet = true",
                "write_manifest = false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = build_prep_config(config_path=config_path)

    assert cfg.source_db == (tmp_path / "../data/anilist.sqlite").resolve()
    assert cfg.semantic_ids_path == (tmp_path / "../data/semantic_ids.jsonl").resolve()
    assert cfg.out_dir == (tmp_path / "./out").resolve()
    assert cfg.rebuild is False
    assert cfg.seed == 123
    assert cfg.target_examples == 99
    assert cfg.split_task_a == 0.5
    assert cfg.split_task_b == 0.25
    assert cfg.split_task_c == 0.25
    assert cfg.max_task_a_templates_per_anime == 10
    assert cfg.task_b_min_history == 4
    assert cfg.task_b_max_history == 20
    assert cfg.task_b_mask_modes == ["last"]
    assert cfg.task_b_positive_score_min == 8
    assert cfg.task_b_allowed_statuses == [2, 4]
    assert cfg.export_parquet is True
    assert cfg.write_manifest is False


def test_build_prep_config_rejects_invalid_split_sum(tmp_path: Path) -> None:
    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                "split_task_a = 0.6",
                "split_task_b = 0.2",
                "split_task_c = 0.1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        build_prep_config(config_path=config_path)


def test_build_prep_config_rejects_invalid_mask_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                'task_b_mask_modes = ["middle"]',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        build_prep_config(config_path=config_path)


def test_build_train_config_resolves_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                "",
                "[train]",
                'base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"',
                'dry_run_model_name = "sshleifer/tiny-gpt2"',
                'train_jsonl = "./llm.jsonl"',
                'out_dir = "./train_out"',
                'env_file = "./.env"',
                'device = "auto"',
                'precision = "bf16"',
                "checkpoint_every_steps = 500",
                "dry_run = true",
                "dry_run_max_steps = 10",
                "dry_run_sample_rows = 2048",
                'dry_run_out_subdir = "dry"',
                "",
                "[train.lora]",
                "enabled = true",
                "r = 8",
                "alpha = 16",
                "dropout = 0.1",
                'target_modules = ["q_proj", "v_proj"]',
                "",
                "[train.tokens]",
                'semantic_vocab_path = "./semantic_vocab.json"',
                "add_special_tokens = true",
                "",
                "[train.tokens.warm_start]",
                "enabled = true",
                'semantic_ids_path = "./semantic_ids.jsonl"',
                'rqvae_checkpoint_path = "./rqvae.pt"',
                "ridge_lambda = 0.001",
                "max_fit_samples = 1000",
                "",
                "[train.phase1]",
                "epochs = 2",
                "weight_task_a = 0.85",
                "weight_task_b = 0.10",
                "weight_task_c = 0.05",
                "max_domain_rows = 0",
                "",
                "[train.phase2]",
                "epochs = 3",
                "weight_task_a = 0.10",
                "weight_task_b = 0.45",
                "weight_task_c = 0.45",
                "max_domain_rows = 0",
                "",
                "[train.general_mix]",
                "ratio = 0.15",
                'sources = ["SlimOrca", "OpenHermes"]',
                "max_rows_per_source = 1000",
                'cache_dir = "./hf_cache"',
                "seed = 42",
                "",
                "[train.eval]",
                'english_eval_dataset = "wikitext2"',
                "eval_every_steps = 500",
                "eval_max_samples = 128",
                "eval_batch_size = 2",
                "",
                "[train.wandb]",
                "enabled = false",
                'mode = "offline"',
                'project = "proj"',
                'entity = "ent"',
                'api_key = ""',
                'run_name = "run"',
                'group = "grp"',
                'tags = ["a"]',
                'project_env = "WANDB_PROJECT"',
                'entity_env = "WANDB_ENTITY"',
                'api_key_env = "WANDB_API_KEY"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = build_train_config(config_path=config_path)
    assert cfg.dry_run_model_name == "sshleifer/tiny-gpt2"
    assert cfg.train_jsonl == (tmp_path / "./llm.jsonl").resolve()
    assert cfg.out_dir == (tmp_path / "./train_out").resolve()
    assert cfg.tokens.semantic_vocab_path == (tmp_path / "./semantic_vocab.json").resolve()
    assert cfg.tokens.warm_start.semantic_ids_path == (tmp_path / "./semantic_ids.jsonl").resolve()
    assert cfg.tokens.warm_start.rqvae_checkpoint_path == (tmp_path / "./rqvae.pt").resolve()
    assert cfg.lora.target_modules == ["q_proj", "v_proj"]
    assert cfg.general_mix.sources == ["SlimOrca", "OpenHermes"]
    assert cfg.eval.english_eval_dataset == "wikitext2"


def test_build_train_config_rejects_bad_phase_weights(tmp_path: Path) -> None:
    config_path = tmp_path / "recommender.toml"
    config_path.write_text(
        "\n".join(
            [
                "[prep]",
                "",
                "[train]",
                "",
                "[train.lora]",
                "",
                "[train.tokens]",
                "",
                "[train.tokens.warm_start]",
                "",
                "[train.phase1]",
                "epochs = 2",
                "weight_task_a = 0.9",
                "weight_task_b = 0.2",
                "weight_task_c = 0.1",
                "",
                "[train.phase2]",
                "epochs = 3",
                "weight_task_a = 0.1",
                "weight_task_b = 0.45",
                "weight_task_c = 0.45",
                "",
                "[train.general_mix]",
                "ratio = 0.15",
                'sources = ["SlimOrca"]',
                "",
                "[train.eval]",
                'english_eval_dataset = "wikitext2"',
                "",
                "[train.wandb]",
                'mode = "offline"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        build_train_config(config_path=config_path)
