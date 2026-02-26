from __future__ import annotations

from pathlib import Path

from animind_tokenizer.config import build_embed_config, build_prep_config


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
                "rebuild = true",
                "limit = 321",
                'model_name = "test/model"',
                "batch_size = 16",
                "max_length = 1024",
                'device = "cuda"',
                "normalize = false",
            ]
        ),
        encoding="utf-8",
    )

    prep_cfg = build_prep_config(config_path=config_path)
    embed_cfg = build_embed_config(config_path=config_path)

    assert prep_cfg.source_db == (tmp_path / "../output/anilist.sqlite").resolve()
    assert prep_cfg.out_dir == (tmp_path / "./output").resolve()
    assert prep_cfg.rebuild is False
    assert prep_cfg.export_parquet is False
    assert prep_cfg.limit == 123

    assert embed_cfg.tokenizer_db == (tmp_path / "./output/tokenizer.sqlite").resolve()
    assert embed_cfg.rebuild is True
    assert embed_cfg.limit == 321
    assert embed_cfg.model_name == "test/model"
    assert embed_cfg.batch_size == 16
    assert embed_cfg.max_length == 1024
    assert embed_cfg.device == "cuda"
    assert embed_cfg.normalize is False


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

