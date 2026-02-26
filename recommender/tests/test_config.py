from __future__ import annotations

from pathlib import Path

import pytest

from animind_recommender.config import build_prep_config


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
