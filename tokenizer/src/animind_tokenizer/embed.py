from __future__ import annotations

import sqlite3
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console

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
    normalize: bool = True


class HuggingFaceEmbeddingBackend:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        max_length: int,
        device: str,
        normalize: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding dependencies are missing. Install torch and transformers to run --phase embed."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.device = _resolve_device(device=device, torch_module=torch)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

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
            return embeddings.detach().cpu().float().tolist()


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

        embed_backend = backend or HuggingFaceEmbeddingBackend(
            model_name=config.model_name,
            batch_size=config.batch_size,
            max_length=config.max_length,
            device=config.device,
            normalize=config.normalize,
        )

        query = "SELECT anime_id, anime_text, prepared_at FROM anime_prep ORDER BY anime_id"
        if config.limit > 0:
            query += f" LIMIT {int(config.limit)}"

        cursor = conn.execute(query)
        total_rows = 0
        rows_written = 0
        embedding_dim: int | None = None

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
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")
    if normalized not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported device: {device}")
    return normalized


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


def _vector_to_blob(vector: list[float]) -> bytes:
    packed = array("f", [float(v) for v in vector])
    return packed.tobytes()


def blob_to_vector(blob: bytes) -> list[float]:
    unpacked = array("f")
    unpacked.frombytes(blob)
    return list(unpacked)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

