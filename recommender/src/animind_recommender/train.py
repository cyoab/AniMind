from __future__ import annotations

import json
import inspect
import random
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .train_data import (
    build_stage_examples,
    format_instruction_pair,
    load_domain_rows,
    load_general_instruction_rows,
    split_train_eval,
    summarize_task_counts,
)
from .train_eval import compute_english_perplexity, load_english_eval_texts
from .train_runtime import cuda_unavailable_reason, resolve_device, resolve_torch_dtype, verify_runtime
from .train_tokens import TokenExtensionResult, extend_tokenizer_and_embeddings
from .train_types import TrainConfig
from .train_wandb import finish_wandb, init_wandb, load_dotenv, resolve_env_value, wandb_log


@dataclass(slots=True)
class StageResult:
    stage: str
    train_rows: int
    eval_rows: int
    metrics: dict[str, Any]
    checkpoint_dir: str
    best_model_dir: str
    english_ppl_samples: list[dict[str, Any]]


def run_train(config: TrainConfig) -> None:
    console = Console()
    effective_config = _apply_dry_run_overrides(config=config, console=console)
    out_dir = effective_config.effective_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not effective_config.train_jsonl.exists():
        raise RuntimeError(f"Train dataset file not found: {effective_config.train_jsonl}")
    if not effective_config.tokens.semantic_vocab_path.exists():
        raise RuntimeError(f"Semantic vocab file not found: {effective_config.tokens.semantic_vocab_path}")

    try:
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install recommender training deps with uv."
        ) from exc

    set_seed(effective_config.seed)
    random.seed(effective_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_config.seed)

    requested_device = effective_config.device.strip().lower()
    resolved_device = resolve_device(device=effective_config.device, torch_module=torch)
    if requested_device == "auto" and resolved_device == "cpu":
        console.log(
            "[yellow]CUDA unavailable; falling back to CPU.[/yellow] "
            f"{cuda_unavailable_reason(torch_module=torch)}"
        )
    effective_precision = effective_config.precision
    if resolved_device == "cpu" and str(effective_precision).lower() in {
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
    }:
        console.log(
            "[yellow]CPU fallback detected; overriding precision to float32 for compatibility.[/yellow]"
        )
        effective_precision = "float32"
    dtype = resolve_torch_dtype(
        device=resolved_device,
        precision=effective_precision,
        torch_module=torch,
    )
    model_name = effective_config.base_model
    if effective_config.dry_run and effective_config.dry_run_model_name.strip():
        model_name = effective_config.dry_run_model_name.strip()
        console.log(
            f"[yellow]Dry-run model override enabled:[/yellow] using {model_name} "
            f"(configured base_model={effective_config.base_model})."
        )

    hf_token = _resolve_hf_token(
        env_file=effective_config.env_file,
        direct_token=effective_config.hf_token,
        env_key=effective_config.hf_token_env,
    )
    token_arg = hf_token or None

    console.rule("[bold cyan]LLM Training[/bold cyan]")
    stage_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*stage_columns, console=console) as stage_progress:
        stage_task = stage_progress.add_task("[cyan]Train stages[/cyan]", total=9)
        fallback_local_model = False

        stage_progress.update(stage_task, description="[cyan]1/9 Load model tokenizer[/cyan]")
        try:
            with console.status("[cyan]Retrieving tokenizer (hub/cache)...[/cyan]", spinner="dots"):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=token_arg,
                )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            with console.status("[cyan]Retrieving model weights (hub/cache)...[/cyan]", spinner="dots"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=token_arg,
                    torch_dtype=dtype,
                )
        except Exception as exc:
            if not effective_config.dry_run:
                raise
            fallback_local_model = True
            model_name = "__local_dryrun_gpt2__"
            console.log(
                "[yellow]Model retrieval failed in dry-run; using local fallback tiny model/tokenizer.[/yellow] "
                f"{exc}"
            )
            tokenizer, model = _build_local_dryrun_model_and_tokenizer(
                train_jsonl=effective_config.train_jsonl,
                max_seq_len=effective_config.max_seq_len,
                torch_module=torch,
            )
        model.to(resolved_device)
        gpu_name, gpu_vram = verify_runtime(
            resolved_device=resolved_device,
            dtype=dtype,
            torch_module=torch,
        )
        stage_progress.advance(stage_task)
        console.print(
            _build_runtime_table(
                config=effective_config,
                model_name=model_name,
                resolved_device=resolved_device,
                dtype=dtype,
                gpu_name=gpu_name,
                gpu_vram=gpu_vram,
            )
        )

        stage_progress.update(stage_task, description="[cyan]2/9 Extend tokenizer and embeddings[/cyan]")
        token_result = extend_tokenizer_and_embeddings(
            tokenizer=tokenizer,
            model=model,
            cfg=effective_config.tokens,
            console=console,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]3/9 Configure LoRA/runtime[/cyan]")
        if effective_config.lora.enabled:
            if fallback_local_model:
                console.log(
                    "[yellow]Skipping LoRA in local dry-run fallback model.[/yellow]"
                )
            else:
                lora_cfg = PeftLoraConfig(
                    r=effective_config.lora.r,
                    lora_alpha=effective_config.lora.alpha,
                    lora_dropout=effective_config.lora.dropout,
                    target_modules=effective_config.lora.target_modules,
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_cfg)
                trainable, total = _count_trainable_params(model)
                console.log(
                    "LoRA enabled: "
                    f"trainable_params={trainable:,}, total_params={total:,}, "
                    f"ratio={(trainable / max(1, total)):.6f}"
                )
        if effective_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]4/9 Load domain/general datasets[/cyan]")
        domain_rows = load_domain_rows(path=effective_config.train_jsonl)
        try:
            general_rows = load_general_instruction_rows(config=effective_config.general_mix, console=console)
        except Exception as exc:
            if not effective_config.dry_run:
                raise
            console.log(
                "[yellow]General mix dataset load failed in dry-run; using synthetic fallback rows.[/yellow] "
                f"{exc}"
            )
            general_rows = _build_synthetic_general_rows(domain_rows=domain_rows, max_rows=4096)
        try:
            english_eval_texts = load_english_eval_texts(
                dataset_name=effective_config.eval.english_eval_dataset,
                max_samples=effective_config.eval.eval_max_samples,
                cache_dir=str(effective_config.general_mix.cache_dir),
            )
        except Exception as exc:
            if not effective_config.dry_run:
                raise
            console.log(
                "[yellow]English eval dataset load failed in dry-run; using local fallback texts.[/yellow] "
                f"{exc}"
            )
            english_eval_texts = _default_english_eval_texts()
        stage_progress.advance(stage_task)
        console.log(
            "Dataset pools ready: "
            f"domain_A={len(domain_rows['A']):,}, domain_B={len(domain_rows['B']):,}, "
            f"domain_C={len(domain_rows['C']):,}, general={len(general_rows):,}, "
            f"english_eval={len(english_eval_texts):,}."
        )

        stage_progress.update(stage_task, description="[cyan]5/9 Build stage-1 datasets[/cyan]")
        stage1_rows = build_stage_examples(
            phase_name="phase1",
            phase_cfg=effective_config.phase1,
            domain_rows=domain_rows,
            general_rows=general_rows,
            general_ratio=effective_config.general_mix.ratio,
            seed=effective_config.seed,
        )
        stage1_train_rows, stage1_eval_rows = split_train_eval(
            rows=stage1_rows,
            eval_ratio=0.02,
            seed=effective_config.seed,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]6/9 Build stage-2 datasets[/cyan]")
        stage2_rows = build_stage_examples(
            phase_name="phase2",
            phase_cfg=effective_config.phase2,
            domain_rows=domain_rows,
            general_rows=general_rows,
            general_ratio=effective_config.general_mix.ratio,
            seed=effective_config.seed + 1,
        )
        stage2_train_rows, stage2_eval_rows = split_train_eval(
            rows=stage2_rows,
            eval_ratio=0.02,
            seed=effective_config.seed + 1,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]7/9 Tokenize stage datasets[/cyan]")
        train_ds_1 = _build_tokenized_dataset(
            rows=stage1_train_rows,
            tokenizer=tokenizer,
            max_seq_len=effective_config.max_seq_len,
        )
        eval_ds_1 = _build_tokenized_dataset(
            rows=stage1_eval_rows,
            tokenizer=tokenizer,
            max_seq_len=effective_config.max_seq_len,
        )
        train_ds_2 = _build_tokenized_dataset(
            rows=stage2_train_rows,
            tokenizer=tokenizer,
            max_seq_len=effective_config.max_seq_len,
        )
        eval_ds_2 = _build_tokenized_dataset(
            rows=stage2_eval_rows,
            tokenizer=tokenizer,
            max_seq_len=effective_config.max_seq_len,
        )
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]8/9 Train stage 1 + stage 2[/cyan]")
        wandb_run = init_wandb(
            cfg=effective_config.wandb,
            env_file=effective_config.env_file,
            metadata={
                "base_model": effective_config.base_model,
                "effective_model_name": model_name,
                "precision": effective_precision,
                "seed": effective_config.seed,
                "dry_run": effective_config.dry_run,
                "checkpoint_every_steps": effective_config.checkpoint_every_steps,
                "general_mix_ratio": effective_config.general_mix.ratio,
            },
            console=console,
        )
        try:
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            stage1_result = _run_training_stage(
                stage_name="phase1_linking",
                stage_index=1,
                config=effective_config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds_1,
                eval_dataset=eval_ds_1,
                output_dir=out_dir / "phase1_linking",
                collator=collator,
                english_eval_texts=english_eval_texts,
                resolved_device=resolved_device,
                torch_module=torch,
                trainer_types={
                    "Trainer": Trainer,
                    "TrainingArguments": TrainingArguments,
                    "TrainerCallback": TrainerCallback,
                },
                wandb_run=wandb_run,
                console=console,
                resume_from=effective_config.resume_from or None,
                stage_epochs=effective_config.phase1.epochs,
            )
            stage2_result = _run_training_stage(
                stage_name="phase2_reasoning",
                stage_index=2,
                config=effective_config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds_2,
                eval_dataset=eval_ds_2,
                output_dir=out_dir / "phase2_reasoning",
                collator=collator,
                english_eval_texts=english_eval_texts,
                resolved_device=resolved_device,
                torch_module=torch,
                trainer_types={
                    "Trainer": Trainer,
                    "TrainingArguments": TrainingArguments,
                    "TrainerCallback": TrainerCallback,
                },
                wandb_run=wandb_run,
                console=console,
                resume_from=None,
                stage_epochs=effective_config.phase2.epochs,
            )
        finally:
            finish_wandb(run=wandb_run)
        stage_progress.advance(stage_task)

        stage_progress.update(stage_task, description="[cyan]9/9 Save final artifacts[/cyan]")
        final_model_dir = out_dir / "final_model"
        final_tokenizer_dir = out_dir / "tokenizer"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        final_tokenizer_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_tokenizer_dir)
        summary_payload = _build_summary_payload(
            config=effective_config,
            model_name=model_name,
            token_result=token_result,
            stage_results=[stage1_result, stage2_result],
            stage_row_stats={
                "stage1_train_task_counts": summarize_task_counts(stage1_train_rows),
                "stage1_eval_task_counts": summarize_task_counts(stage1_eval_rows),
                "stage2_train_task_counts": summarize_task_counts(stage2_train_rows),
                "stage2_eval_task_counts": summarize_task_counts(stage2_eval_rows),
            },
            final_model_dir=final_model_dir,
            final_tokenizer_dir=final_tokenizer_dir,
        )
        summary_path = out_dir / "train_run_summary.json"
        _write_json(summary_path, summary_payload)
        _write_json(out_dir / "train_config_snapshot.json", _config_as_json(config=effective_config))
        if effective_config.dry_run:
            _validate_dry_run_artifacts(out_dir=out_dir)
        stage_progress.advance(stage_task)

    console.log(
        "Train complete: "
        f"out_dir={out_dir}, "
        f"final_model={out_dir / 'final_model'}, "
        f"summary={out_dir / 'train_run_summary.json'}."
    )


def _run_training_stage(
    *,
    stage_name: str,
    stage_index: int,
    config: TrainConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: Path,
    collator: Any,
    english_eval_texts: list[str],
    resolved_device: str,
    torch_module: Any,
    trainer_types: dict[str, Any],
    wandb_run: Any | None,
    console: Console,
    resume_from: str | None,
    stage_epochs: int,
) -> StageResult:
    Trainer = trainer_types["Trainer"]
    TrainingArguments = trainer_types["TrainingArguments"]
    TrainerCallback = trainer_types["TrainerCallback"]

    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = config.dry_run_max_steps if config.dry_run else -1
    save_steps = config.checkpoint_every_steps
    eval_steps = config.eval.eval_every_steps
    if config.dry_run:
        save_steps = max(1, min(save_steps, config.dry_run_max_steps))
        eval_steps = max(1, min(eval_steps, config.dry_run_max_steps))

    class StageCallback(TrainerCallback):
        def __init__(self) -> None:
            self.english_samples: list[dict[str, Any]] = []

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step <= 0 or state.global_step % eval_steps != 0:
                return control
            model_ref = kwargs.get("model")
            if model_ref is None:
                return control
            ppl = compute_english_perplexity(
                model=model_ref,
                tokenizer=tokenizer,
                texts=english_eval_texts,
                max_seq_len=config.max_seq_len,
                batch_size=config.eval.eval_batch_size,
                device=resolved_device,
                torch_module=torch_module,
            )
            sample = {"step": int(state.global_step), "english_ppl": float(ppl)}
            self.english_samples.append(sample)
            wandb_log(
                run=wandb_run,
                payload={f"{stage_name}/english_ppl": float(ppl)},
                step=int(state.global_step),
            )
            console.log(f"{stage_name}: english_ppl@{state.global_step}={ppl:.6f}")
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            if not logs:
                return control
            payload = {f"{stage_name}/{key}": value for key, value in logs.items()}
            wandb_log(run=wandb_run, payload=payload, step=int(state.global_step))
            return control

    callback = StageCallback()
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(stage_epochs),
        "max_steps": int(max_steps),
        "per_device_train_batch_size": int(config.per_device_train_batch_size),
        "per_device_eval_batch_size": int(config.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "warmup_ratio": float(config.warmup_ratio),
        "max_grad_norm": float(config.max_grad_norm),
        "logging_steps": max(1, min(50, save_steps)),
        "save_strategy": "steps",
        "save_steps": int(save_steps),
        "save_total_limit": int(config.save_total_limit),
        "eval_steps": int(eval_steps),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": bool(resolved_device.startswith("cuda") and str(config.precision).lower() in {"fp16", "float16"}),
        "bf16": bool(resolved_device.startswith("cuda") and str(config.precision).lower() in {"bf16", "bfloat16"}),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    if "eval_strategy" in params:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        training_kwargs["evaluation_strategy"] = "steps"
    if "overwrite_output_dir" in params:
        training_kwargs["overwrite_output_dir"] = bool(config.rebuild)

    filtered_kwargs = {key: value for key, value in training_kwargs.items() if key in params}
    training_args = TrainingArguments(**filtered_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[callback],
    )

    resume_checkpoint = None
    if resume_from:
        resume_checkpoint = str(resume_from).strip()
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    metrics.update({f"eval_{key}": value for key, value in eval_metrics.items() if key not in metrics})
    if not callback.english_samples:
        ppl = compute_english_perplexity(
            model=trainer.model,
            tokenizer=tokenizer,
            texts=english_eval_texts,
            max_seq_len=config.max_seq_len,
            batch_size=config.eval.eval_batch_size,
            device=resolved_device,
            torch_module=torch_module,
        )
        callback.english_samples.append({"step": int(getattr(trainer.state, "global_step", 0)), "english_ppl": float(ppl)})
        wandb_log(
            run=wandb_run,
            payload={f"{stage_name}/english_ppl": float(ppl)},
            step=int(getattr(trainer.state, "global_step", 0)),
        )

    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir / "tokenizer"))

    checkpoint_dir = str(output_dir)
    return StageResult(
        stage=stage_name,
        train_rows=len(train_dataset),
        eval_rows=len(eval_dataset),
        metrics=metrics,
        checkpoint_dir=checkpoint_dir,
        best_model_dir=str(best_model_dir),
        english_ppl_samples=callback.english_samples,
    )


def _build_tokenized_dataset(*, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> Any:
    if not rows:
        raise RuntimeError("Cannot tokenize empty rows.")
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Tokenization requires datasets package. Install with: uv add datasets"
        ) from exc

    texts = [
        format_instruction_pair(input_text=str(row["input"]), output_text=str(row["output"]))
        for row in rows
    ]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        return tokenized

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized


def _resolve_hf_token(*, env_file: Path, direct_token: str, env_key: str) -> str:
    dotenv_values = load_dotenv(env_file)
    return resolve_env_value(
        direct_value=direct_token,
        env_key=env_key,
        dotenv_values=dotenv_values,
    )


def _count_trainable_params(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        value = int(parameter.numel())
        total += value
        if bool(parameter.requires_grad):
            trainable += value
    return trainable, total


def _build_runtime_table(
    *,
    config: TrainConfig,
    model_name: str,
    resolved_device: str,
    dtype: Any,
    gpu_name: str | None,
    gpu_vram: float | None,
) -> Table:
    table = Table(title="Train Runtime Summary", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Base Model", model_name)
    table.add_row("Device", resolved_device)
    table.add_row("Precision", str(dtype).replace("torch.", ""))
    table.add_row("Seq Len", str(config.max_seq_len))
    table.add_row("Batch / Grad Accum", f"{config.per_device_train_batch_size} / {config.gradient_accumulation_steps}")
    table.add_row("Checkpoint Every", str(config.checkpoint_every_steps))
    table.add_row("Dry Run", str(config.dry_run))
    if gpu_name is not None and gpu_vram is not None:
        table.add_row("GPU", f"{gpu_name} ({gpu_vram:.2f} GiB)")
    return table


def _apply_dry_run_overrides(*, config: TrainConfig, console: Console) -> TrainConfig:
    if not config.dry_run:
        return config
    dry_phase1 = replace(
        config.phase1,
        epochs=1,
        max_domain_rows=min(max(config.phase1.max_domain_rows, 0) or config.dry_run_sample_rows, config.dry_run_sample_rows),
    )
    dry_phase2 = replace(
        config.phase2,
        epochs=1,
        max_domain_rows=min(max(config.phase2.max_domain_rows, 0) or config.dry_run_sample_rows, config.dry_run_sample_rows),
    )
    dry_cfg = replace(
        config,
        checkpoint_every_steps=max(1, min(config.checkpoint_every_steps, config.dry_run_max_steps)),
        phase1=dry_phase1,
        phase2=dry_phase2,
    )
    console.log(
        "[yellow]Dry-run mode enabled.[/yellow] "
        f"max_steps={config.dry_run_max_steps}, sample_rows={config.dry_run_sample_rows}, out_dir={dry_cfg.effective_out_dir}."
    )
    return dry_cfg


def _build_summary_payload(
    *,
    config: TrainConfig,
    model_name: str,
    token_result: TokenExtensionResult,
    stage_results: list[StageResult],
    stage_row_stats: dict[str, Any],
    final_model_dir: Path,
    final_tokenizer_dir: Path,
) -> dict[str, Any]:
    return {
        "completed_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "effective_model_name": model_name,
        "config": _config_as_json(config=config),
        "token_extension": asdict(token_result),
        "stages": [asdict(stage_result) for stage_result in stage_results],
        "stage_row_stats": stage_row_stats,
        "final_model_dir": str(final_model_dir),
        "final_tokenizer_dir": str(final_tokenizer_dir),
    }


def _config_as_json(*, config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    path_like_keys = {
        "train_jsonl",
        "out_dir",
        "env_file",
        "semantic_vocab_path",
        "semantic_ids_path",
        "rqvae_checkpoint_path",
        "cache_dir",
    }

    def convert(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: convert(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v, key) for v in value]
        if isinstance(value, Path) or key in path_like_keys:
            return str(value)
        return value

    return convert(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def _validate_dry_run_artifacts(*, out_dir: Path) -> None:
    required_paths = (
        out_dir / "phase1_linking",
        out_dir / "phase2_reasoning",
        out_dir / "train_run_summary.json",
        out_dir / "train_config_snapshot.json",
        out_dir / "final_model",
        out_dir / "tokenizer",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Dry-run validation failed; missing artifacts: {', '.join(missing)}")


def _build_synthetic_general_rows(
    *,
    domain_rows: dict[str, list[dict[str, Any]]],
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ("A", "B", "C"):
        for row in domain_rows.get(task, []):
            rows.append(
                {
                    "task": "GEN",
                    "input": f"Explain in plain English: {row['input']}",
                    "output": row["output"],
                    "source": "synthetic_fallback",
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def _default_english_eval_texts() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Language models should preserve fluent English while learning domain tokens.",
        "Anime recommendation benefits from both structured IDs and natural language reasoning.",
        "A reliable training pipeline needs checkpoints, metrics, and reproducible configuration.",
    ]


def _build_local_dryrun_model_and_tokenizer(
    *,
    train_jsonl: Path,
    max_seq_len: int,
    torch_module: Any,
) -> tuple[Any, Any]:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import WordLevelTrainer
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Local dry-run fallback requires tokenizers + transformers packages."
        ) from exc

    bootstrap_texts: list[str] = []
    with train_jsonl.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            input_text = str(row.get("input", "")).strip()
            output_text = str(row.get("output", "")).strip()
            if input_text:
                bootstrap_texts.append(input_text)
            if output_text:
                bootstrap_texts.append(output_text)
            if len(bootstrap_texts) >= 4_096:
                break
    if not bootstrap_texts:
        bootstrap_texts = [
            "hello world",
            "dry run fallback tokenizer",
            "anime recommendation semantic id",
        ]

    tokenizer_obj = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer_obj.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        vocab_size=8_192,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer_obj.train_from_iterator(bootstrap_texts, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=max(128, min(max_seq_len, 1024)),
        n_ctx=max(128, min(max_seq_len, 1024)),
        n_embd=192,
        n_layer=2,
        n_head=3,
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    model = GPT2LMHeadModel(config)
    model.to(torch_module.float32)
    return tokenizer, model
