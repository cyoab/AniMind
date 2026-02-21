from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from animind_dataset import cli as cli_module
from animind_dataset.pipeline import run_build as real_run_build
from tests.fakes import FakeClientConfig, FakeJikanClient


runner = CliRunner()


def test_cli_build_and_stats_smoke(tmp_path: Path, monkeypatch) -> None:
    async def patched_run_build(**kwargs):
        return await real_run_build(
            **kwargs,
            client=FakeJikanClient(FakeClientConfig(user_count=30)),
        )

    monkeypatch.setattr(cli_module, "run_build", patched_run_build)

    out_dir = tmp_path / "out"
    build_result = runner.invoke(
        cli_module.app,
        [
            "build",
            "--target-users",
            "20",
            "--out-dir",
            str(out_dir),
            "--include-nsfw",
        ],
    )
    assert build_result.exit_code == 0, build_result.stdout

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["run"]["run_id"]

    stats_result = runner.invoke(
        cli_module.app,
        [
            "stats",
            "--run-id",
            run_id,
            "--out-dir",
            str(out_dir),
        ],
    )
    assert stats_result.exit_code == 0, stats_result.stdout
    assert "Users discovered" in stats_result.stdout
