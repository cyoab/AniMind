from __future__ import annotations

import pytest
from rich.console import Console

from animind_recommender.train import _resolve_lora_target_modules


class _WeightStub:
    ndim = 2


class _ModuleStub:
    def __init__(self) -> None:
        self.weight = _WeightStub()


class _FakeModel:
    def __init__(self, module_names: list[str]) -> None:
        self._module_names = module_names

    def named_modules(self):
        yield "", object()
        for module_name in self._module_names:
            yield module_name, _ModuleStub()


def test_resolve_lora_target_modules_raises_for_missing_non_dry_run() -> None:
    model = _FakeModel(
        [
            "transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_proj",
        ]
    )
    with pytest.raises(ValueError):
        _resolve_lora_target_modules(
            model=model,
            configured_target_modules=["q_proj", "k_proj"],
            dry_run=False,
            console=Console(record=True),
        )


def test_resolve_lora_target_modules_uses_fallback_in_dry_run() -> None:
    model = _FakeModel(
        [
            "transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_proj",
            "transformer.h.0.mlp.c_fc",
        ]
    )
    resolved = _resolve_lora_target_modules(
        model=model,
        configured_target_modules=["q_proj", "k_proj", "v_proj"],
        dry_run=True,
        console=Console(record=True),
    )
    assert resolved == ["c_attn", "c_proj", "c_fc"]
