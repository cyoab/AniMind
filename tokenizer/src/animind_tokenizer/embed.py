from __future__ import annotations

import os
import sqlite3
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any, Protocol

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

DEFAULT_EMBED_MODEL = "tencent/KaLM-Embedding-Gemma3-12B-2511"


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass(slots=True)
class EmbedConfig:
    tokenizer_db: Path = Path("./output/tokenizer.sqlite")
    rebuild: bool = True
    limit: int = 0
    model_name: str = DEFAULT_EMBED_MODEL
    batch_size: int = 8
    max_length: int = 2048
    device: str = "auto"
    precision: str = "auto"
    env_file: Path = Path("./.env")
    hf_token: str = ""
    hf_token_env: str = "HF_TOKEN"
    normalize: bool = True


class HuggingFaceEmbeddingBackend:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        max_length: int,
        device: str,
        precision: str,
        env_file: Path,
        hf_token: str,
        hf_token_env: str,
        normalize: bool,
        console: Console | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding dependencies are missing. Install torch and transformers to run --phase embed."
            ) from exc

        self._torch = torch
        self.console = console or Console()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.requested_device = device.strip().lower()
        self.requested_precision = precision.strip().lower()
        self.env_file = env_file
        self.hf_token_env = hf_token_env.strip() or "HF_TOKEN"
        self.hf_token = _resolve_hf_token(
            env_file=self.env_file,
            direct_token=hf_token,
            env_key=self.hf_token_env,
        )
        self.hf_token_set = bool(self.hf_token)
        self.device = _resolve_device(device=device, torch_module=torch)
        self.dtype = _resolve_torch_dtype(
            device=self.device,
            precision=self.requested_precision,
            torch_module=torch,
        )
        self.model_retrieved = False
        self.model_loaded = False
        self.gpu_name: str | None = None
        self.gpu_total_memory_gib: float | None = None

        self.console.log(
            "Embedding backend setup: "
            f"model={self.model_name}, requested_device={self.requested_device}, "
            f"resolved_device={self.device}, requested_precision={self.requested_precision}, "
            f"dtype={str(self.dtype).replace('torch.', '')}, "
            f"hf_token={'set' if self.hf_token_set else 'unset'}."
        )
        if self.requested_device == "auto" and self.device == "cpu":
            self.console.log(
                "[yellow]CUDA unavailable; falling back to CPU.[/yellow] "
                f"{_cuda_unavailable_reason(torch_module=torch)}"
            )

        with self.console.status("[cyan]Retrieving tokenizer (hub/cache)...", spinner="dots"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=(self.hf_token or None),
            )
        self.console.log("[green]Tokenizer ready.[/green]")

        with self.console.status("[cyan]Retrieving model weights (hub/cache)...", spinner="dots"):
            self.model = _load_hf_model(
                auto_model_cls=AutoModel,
                model_name=self.model_name,
                dtype=self.dtype,
                token=(self.hf_token or None),
            )
            self.model_retrieved = True
        self.console.log("[green]Model weights retrieved.[/green]")

        with self.console.status(
            f"[cyan]Loading model onto {self.device} and verifying runtime...[/cyan]",
            spinner="dots",
        ):
            self.model.to(self.device)
            self.model.eval()
            self._verify_runtime()
            self.model_loaded = True
        self.console.log(
            "[green]Model loaded and verified.[/green] "
            f"device={self.device}"
            + (
                f", gpu={self.gpu_name}, vram={self.gpu_total_memory_gib:.2f} GiB."
                if self.gpu_name and self.gpu_total_memory_gib is not None
                else "."
            )
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = _extract_embeddings(outputs=outputs, attention_mask=encoded.get("attention_mask"), torch_module=torch)
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            finite_mask = torch.isfinite(embeddings)
            if not bool(finite_mask.all()):
                bad_values = int((~finite_mask).sum().item())
                raise RuntimeError(
                    "Embedding backend produced non-finite values. "
                    f"bad_values={bad_values}, device={self.device}, dtype={str(self.dtype).replace('torch.', '')}. "
                    "This often indicates an unstable model precision/runtime configuration."
                )
            return embeddings.detach().cpu().float().tolist()

    def runtime_summary(self) -> str:
        summary = (
            f"model={self.model_name}, device={self.device}, "
            f"dtype={str(self.dtype).replace('torch.', '')}, "
            f"retrieved={self.model_retrieved}, loaded={self.model_loaded}"
        )
        if self.gpu_name and self.gpu_total_memory_gib is not None:
            summary += f", gpu={self.gpu_name}, vram={self.gpu_total_memory_gib:.2f} GiB"
        return summary

    def _verify_runtime(self) -> None:
        if not self.device.startswith("cuda"):
            return

        try:
            target_device = self._torch.device(self.device)
            cuda_index = (
                int(target_device.index)
                if target_device.index is not None
                else int(self._torch.cuda.current_device())
            )
            props = self._torch.cuda.get_device_properties(cuda_index)
            self.gpu_name = props.name
            self.gpu_total_memory_gib = props.total_memory / (1024**3)

            probe = self._torch.randn((256, 256), device=self.device, dtype=self.dtype)
            _ = (probe @ probe.T).sum().item()
            self._torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - depends on runtime CUDA availability.
            raise RuntimeError(
                "CUDA runtime probe failed while loading the model. "
                "Check pod GPU passthrough, driver, and CUDA runtime compatibility."
            ) from exc


def run_embed(config: EmbedConfig, backend: EmbeddingBackend | None = None) -> None:
    console = Console()
    if not config.tokenizer_db.exists():
        raise RuntimeError(f"Tokenizer DB not found: {config.tokenizer_db}")

    with sqlite3.connect(config.tokenizer_db) as conn:
        conn.execute("PRAGMA busy_timeout = 8000;")

        if not _table_exists(conn, "anime_prep"):
            raise RuntimeError(
                "Missing anime_prep table. Run --phase prep before --phase embed."
            )

        if not config.rebuild and _table_exists(conn, "anime_embeddings"):
            existing_rows = _table_row_count(conn, "anime_embeddings")
            if existing_rows > 0:
                console.log(
                    "[yellow]Reuse enabled:[/yellow] "
                    f"keeping existing anime_embeddings ({existing_rows:,} rows)."
                )
                return

        _create_embeddings_table(conn=conn, rebuild=config.rebuild)
        planned_rows = _planned_embed_rows(conn=conn, limit=config.limit)

        embed_backend = backend or HuggingFaceEmbeddingBackend(
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=config.device,
            precision=config.precision,
            env_file=config.env_file,
            hf_token=config.hf_token,
            hf_token_env=config.hf_token_env,
            normalize=config.normalize,
            console=console,
        )
        if hasattr(embed_backend, "runtime_summary"):
            runtime_summary = str(embed_backend.runtime_summary())
            console.log(f"Embedding runtime ready: {runtime_summary}")

        query = "SELECT anime_id, anime_text, prepared_at FROM anime_prep ORDER BY anime_id"
        if config.limit > 0:
            query += f" LIMIT {int(config.limit)}"

        cursor = conn.execute(query)
        total_rows = 0
        rows_written = 0
        embedding_dim: int | None = None

        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        with Progress(*progress_columns, console=console) as progress:
            embed_task = progress.add_task(
                "[cyan]Embedding anime_text rows[/cyan]",
                total=planned_rows,
            )
            while True:
                batch = cursor.fetchmany(config.batch_size)
                if not batch:
                    break
                total_rows += len(batch)
                texts = [str(row[1]) for row in batch]
                vectors = embed_backend.embed_texts(texts)
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding backend returned a mismatched number of vectors.")

                values: list[tuple[Any, ...]] = []
                for row, vector in zip(batch, vectors, strict=True):
                    if _vector_has_non_finite(vector):
                        raise RuntimeError(
                            "Non-finite embedding detected; aborting write to prevent corrupt training data. "
                            f"anime_id={int(row[0])}, model={config.model_name}."
                        )
                    if embedding_dim is None:
                        embedding_dim = len(vector)
                    elif len(vector) != embedding_dim:
                        raise RuntimeError("Inconsistent embedding dimensions returned by backend.")
                    values.append(
                        (
                            int(row[0]),
                            sqlite3.Binary(_vector_to_blob(vector)),
                            int(len(vector)),
                            config.model_name,
                            _now(),
                            row[2],
                        )
                    )

                conn.executemany(
                    """
                    INSERT OR REPLACE INTO anime_embeddings(
                        anime_id, embedding, embedding_dim, model_name, embedded_at, source_prepared_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                conn.commit()
                rows_written += len(values)
                progress.advance(embed_task, advance=len(values))

        console.log(
            "Embed complete: "
            f"input_rows={total_rows:,}, "
            f"rows_written={rows_written:,}, "
            f"embedding_dim={embedding_dim or 0}, "
            f"model={config.model_name}."
        )


def _resolve_device(device: str, torch_module: Any) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if _cuda_runtime_available(torch_module=torch_module) else "cpu"
    if normalized == "cpu":
        return "cpu"

    if normalized == "cuda":
        if not _cuda_runtime_available(torch_module=torch_module):
            raise RuntimeError(
                "CUDA device requested but no usable GPU is available. "
                f"{_cuda_unavailable_reason(torch_module=torch_module)}"
            )
        return "cuda"

    if normalized.startswith("cuda:"):
        raw_index = normalized.split(":", 1)[1]
        if not raw_index.isdigit():
            raise RuntimeError(f"Unsupported CUDA device format: {device}")
        if not _cuda_runtime_available(torch_module=torch_module):
            raise RuntimeError(
                f"CUDA device requested ({normalized}) but no usable GPU is available. "
                f"{_cuda_unavailable_reason(torch_module=torch_module)}"
            )
        index = int(raw_index)
        device_count = _safe_cuda_device_count(torch_module=torch_module)
        if index >= device_count:
            raise RuntimeError(
                f"CUDA device index out of range: requested={index}, visible_devices={device_count}."
            )
        return normalized

    if normalized not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported device: {device}")
    return "cpu"


def _cuda_runtime_available(torch_module: Any) -> bool:
    try:
        if bool(torch_module.cuda.is_available()):
            return True
    except Exception:
        pass
    return _safe_cuda_device_count(torch_module=torch_module) > 0


def _safe_cuda_device_count(torch_module: Any) -> int:
    try:
        return int(torch_module.cuda.device_count())
    except Exception:
        return 0


def _cuda_unavailable_reason(torch_module: Any) -> str:
    parts: list[str] = []
    nvidia_visible_devices = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if nvidia_visible_devices is not None:
        parts.append(f"NVIDIA_VISIBLE_DEVICES={nvidia_visible_devices}")
    if cuda_visible_devices is not None:
        parts.append(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    built_with_cuda = False
    try:
        built_with_cuda = bool(torch_module.backends.cuda.is_built())
    except Exception:
        pass
    parts.append(f"torch_cuda_built={built_with_cuda}")
    parts.append(f"visible_cuda_devices={_safe_cuda_device_count(torch_module=torch_module)}")
    cuinit_code, cuinit_name = _cuinit_probe()
    if cuinit_code is not None:
        parts.append(f"cuInit={cuinit_code}({cuinit_name or 'unknown'})")
    elif cuinit_name:
        parts.append(f"cuInit_probe_error={cuinit_name}")
    return "CUDA diagnostics: " + ", ".join(parts)


def _cuinit_probe() -> tuple[int | None, str | None]:
    try:
        import ctypes

        libcuda = ctypes.CDLL("libcuda.so.1")
        cu_init = libcuda.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        code = int(cu_init(0))

        name: str | None = None
        try:
            get_error_name = libcuda.cuGetErrorName
            get_error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            get_error_name.restype = ctypes.c_int
            raw_name = ctypes.c_char_p()
            get_error_name(code, ctypes.byref(raw_name))
            if raw_name.value is not None:
                name = raw_name.value.decode()
        except Exception:
            name = None
        return code, name
    except Exception as exc:
        return None, str(exc)


def _extract_embeddings(outputs: Any, attention_mask: Any, torch_module: Any) -> Any:
    if hasattr(outputs, "sentence_embeddings") and outputs.sentence_embeddings is not None:
        return outputs.sentence_embeddings
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    hidden = None
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        hidden = outputs[0]
    if hidden is None:
        raise RuntimeError("Could not extract embeddings from model outputs.")

    if hidden.dim() == 2:
        return hidden
    if hidden.dim() != 3:
        raise RuntimeError("Unexpected embedding tensor shape.")
    if attention_mask is None:
        return hidden.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=torch_module.finfo(torch_module.float32).eps)
    return summed / counts


def _create_embeddings_table(conn: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS anime_embeddings")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anime_embeddings (
            anime_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            source_prepared_at TEXT
        )
        """
    )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,)
    ).fetchone()
    return row is not None


def _table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _planned_embed_rows(conn: sqlite3.Connection, limit: int) -> int:
    total_rows = _table_row_count(conn, "anime_prep")
    if limit > 0:
        return min(limit, total_rows)
    return total_rows


def _vector_to_blob(vector: list[float]) -> bytes:
    packed = array("f", [float(v) for v in vector])
    return packed.tobytes()


def blob_to_vector(blob: bytes) -> list[float]:
    unpacked = array("f")
    unpacked.frombytes(blob)
    return list(unpacked)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _vector_has_non_finite(vector: list[float]) -> bool:
    return any(not math.isfinite(float(value)) for value in vector)


def _resolve_torch_dtype(device: str, precision: str, torch_module: Any) -> Any:
    normalized = precision.strip().lower()
    aliases = {
        "auto": "auto",
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    if normalized not in aliases:
        raise RuntimeError(
            "Unsupported embedd.precision value. "
            "Expected one of: auto, float32, float16, bfloat16."
        )
    mode = aliases[normalized]

    if not device.startswith("cuda"):
        if mode in {"float16", "bfloat16"}:
            raise RuntimeError(
                "embedd.precision=float16/bfloat16 requires CUDA device. "
                f"Resolved device is {device}."
            )
        return torch_module.float32

    if mode == "float32":
        return torch_module.float32
    if mode == "float16":
        return torch_module.float16
    if mode == "bfloat16":
        try:
            if hasattr(torch_module.cuda, "is_bf16_supported") and bool(torch_module.cuda.is_bf16_supported()):
                return torch_module.bfloat16
        except Exception:
            pass
        raise RuntimeError(
            "embedd.precision=bfloat16 requested but CUDA BF16 is not supported on this GPU/runtime."
        )

    try:
        if hasattr(torch_module.cuda, "is_bf16_supported") and bool(torch_module.cuda.is_bf16_supported()):
            return torch_module.bfloat16
    except Exception:
        pass
    return torch_module.float32


def _load_hf_model(*, auto_model_cls: Any, model_name: str, dtype: Any, token: str | None) -> Any:
    try:
        return auto_model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
            token=token,
        )
    except TypeError:
        return auto_model_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            token=token,
        )


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _resolve_hf_token(*, env_file: Path, direct_token: str, env_key: str) -> str:
    if direct_token.strip():
        return direct_token.strip()

    dotenv_values = _load_dotenv(env_file)
    token = dotenv_values.get(env_key, "").strip()
    if token:
        return token

    env_token = os.environ.get(env_key, "").strip()
    if env_token:
        return env_token

    # Common Hugging Face default env names.
    for fallback in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        candidate = dotenv_values.get(fallback, "").strip() or os.environ.get(fallback, "").strip()
        if candidate:
            return candidate
    return ""
