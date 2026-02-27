from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass(slots=True)
class TokenWarmStartConfig:
    enabled: bool = False
    semantic_ids_path: Path = Path("../data/semantic_ids/conservative/semantic_ids.jsonl")
    rqvae_checkpoint_path: Path = Path("../../tokenizer/artifacts/rqvae_conservative/rqvae_best_infer.pt")
    ridge_lambda: float = 1e-3
    max_fit_samples: int = 50_000


@dataclass(slots=True)
class TokenExtensionConfig:
    semantic_vocab_path: Path = Path("../data/semantic_ids/conservative/semantic_vocab.json")
    add_special_tokens: bool = True
    warm_start: TokenWarmStartConfig = field(default_factory=TokenWarmStartConfig)


@dataclass(slots=True)
class PhaseConfig:
    epochs: int = 2
    weight_task_a: float = 0.85
    weight_task_b: float = 0.10
    weight_task_c: float = 0.05
    max_domain_rows: int = 0


@dataclass(slots=True)
class GeneralMixConfig:
    ratio: float = 0.15
    sources: list[str] = field(default_factory=lambda: ["SlimOrca", "OpenHermes"])
    max_rows_per_source: int = 60_000
    cache_dir: Path = Path("../output/hf_cache")
    seed: int = 42


@dataclass(slots=True)
class EvalConfig:
    english_eval_dataset: str = "wikitext2"
    eval_every_steps: int = 500
    eval_max_samples: int = 512
    eval_batch_size: int = 2


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = False
    mode: str = "offline"
    project: str = ""
    entity: str = ""
    api_key: str = ""
    run_name: str = ""
    group: str = ""
    tags: list[str] = field(default_factory=list)
    project_env: str = "WANDB_PROJECT"
    entity_env: str = "WANDB_ENTITY"
    api_key_env: str = "WANDB_API_KEY"


@dataclass(slots=True)
class TrainConfig:
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    dry_run_model_name: str = "sshleifer/tiny-gpt2"
    train_jsonl: Path = Path("../output/llm_prep_train.jsonl")
    out_dir: Path = Path("../output/train")
    env_file: Path = Path("../.env")
    rebuild: bool = True
    seed: int = 42
    device: str = "auto"
    precision: str = "bf16"
    max_seq_len: int = 1024
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 500
    save_total_limit: int = 3
    resume_from: str = ""
    hf_token: str = ""
    hf_token_env: str = "HF_TOKEN"
    dry_run: bool = False
    dry_run_max_steps: int = 20
    dry_run_sample_rows: int = 2_048
    dry_run_out_subdir: str = "dry_run"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    tokens: TokenExtensionConfig = field(default_factory=TokenExtensionConfig)
    phase1: PhaseConfig = field(default_factory=PhaseConfig)
    phase2: PhaseConfig = field(
        default_factory=lambda: PhaseConfig(
            epochs=3,
            weight_task_a=0.10,
            weight_task_b=0.45,
            weight_task_c=0.45,
        )
    )
    general_mix: GeneralMixConfig = field(default_factory=GeneralMixConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def effective_out_dir(self) -> Path:
        if not self.dry_run:
            return self.out_dir
        return self.out_dir / self.dry_run_out_subdir
