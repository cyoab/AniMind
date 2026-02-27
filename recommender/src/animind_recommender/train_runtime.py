from __future__ import annotations

import ctypes
import os
from typing import Any


def resolve_device(*, device: str, torch_module: Any) -> str:
    normalized = str(device).strip().lower()
    if not normalized:
        normalized = "auto"

    if normalized == "auto":
        return "cuda" if cuda_runtime_available(torch_module=torch_module) else "cpu"

    if normalized == "cuda":
        if not cuda_runtime_available(torch_module=torch_module):
            raise RuntimeError(
                "train.device=cuda requested but CUDA runtime is unavailable. "
                f"{cuda_unavailable_reason(torch_module=torch_module)}"
            )
        return "cuda"

    if normalized.startswith("cuda:"):
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device format: {device}") from exc
        if not cuda_runtime_available(torch_module=torch_module):
            raise RuntimeError(
                f"train.device={normalized} requested but CUDA runtime is unavailable. "
                f"{cuda_unavailable_reason(torch_module=torch_module)}"
            )
        count = safe_cuda_device_count(torch_module=torch_module)
        if index < 0 or index >= count:
            raise RuntimeError(
                f"Requested CUDA device index {index} is out of range (available={count})."
            )
        return normalized

    if normalized not in {"cpu"}:
        raise RuntimeError(f"Unsupported train.device: {device}. Expected one of auto/cpu/cuda/cuda:<index>.")
    return normalized


def resolve_torch_dtype(*, device: str, precision: str, torch_module: Any) -> Any:
    mapping = {
        "auto": "auto",
        "float32": "float32",
        "fp32": "float32",
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }
    normalized = str(precision).strip().lower()
    if normalized not in mapping:
        raise RuntimeError(
            "Unsupported train.precision. Expected one of: auto, float32, float16, bfloat16."
        )
    mode = mapping[normalized]

    if not str(device).startswith("cuda"):
        if mode in {"float16", "bfloat16"}:
            raise RuntimeError("train.precision=float16/bfloat16 requires CUDA device.")
        return torch_module.float32

    if mode == "float32":
        return torch_module.float32
    if mode == "float16":
        return torch_module.float16
    if mode == "bfloat16":
        if _cuda_bf16_supported(torch_module=torch_module):
            return torch_module.bfloat16
        raise RuntimeError(
            "train.precision=bfloat16 requested but CUDA BF16 is not supported on this runtime."
        )

    if _cuda_bf16_supported(torch_module=torch_module):
        return torch_module.bfloat16
    return torch_module.float16


def verify_runtime(*, resolved_device: str, dtype: Any, torch_module: Any) -> tuple[str | None, float | None]:
    if not resolved_device.startswith("cuda"):
        return None, None
    target_device = torch_module.device(resolved_device)
    cuda_index = (
        int(target_device.index)
        if target_device.index is not None
        else int(torch_module.cuda.current_device())
    )
    props = torch_module.cuda.get_device_properties(cuda_index)
    gpu_name = str(props.name)
    gpu_vram = float(props.total_memory) / (1024**3)
    probe = torch_module.randn((128, 128), device=resolved_device, dtype=dtype)
    _ = (probe @ probe.T).sum().item()
    torch_module.cuda.synchronize()
    return gpu_name, gpu_vram


def cuda_runtime_available(*, torch_module: Any) -> bool:
    try:
        if bool(torch_module.cuda.is_available()):
            return True
    except Exception:
        return False
    return safe_cuda_device_count(torch_module=torch_module) > 0


def safe_cuda_device_count(*, torch_module: Any) -> int:
    try:
        return int(torch_module.cuda.device_count())
    except Exception:
        return 0


def cuda_unavailable_reason(*, torch_module: Any) -> str:
    details: list[str] = [f"torch={getattr(torch_module, '__version__', 'unknown')}"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        details.append(f"CUDA_VISIBLE_DEVICES={visible}")
    try:
        built_with_cuda = bool(torch_module.backends.cuda.is_built())
    except Exception:
        built_with_cuda = False
    details.append(f"torch_cuda_built={built_with_cuda}")
    details.append(f"visible_cuda_devices={safe_cuda_device_count(torch_module=torch_module)}")
    details.append(f"driver={_query_cuda_driver_status()}")
    return "; ".join(details)


def _cuda_bf16_supported(*, torch_module: Any) -> bool:
    try:
        if hasattr(torch_module.cuda, "is_bf16_supported") and bool(torch_module.cuda.is_bf16_supported()):
            return True
    except Exception:
        return False
    return False


def _query_cuda_driver_status() -> str:
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        cu_init = libcuda.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        code = int(cu_init(0))
        if code == 0:
            return "ok"
        try:
            name_fn = libcuda.cuGetErrorName
            name_fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            name_fn.restype = ctypes.c_int
            out = ctypes.c_char_p()
            _ = name_fn(code, ctypes.byref(out))
            return f"error={code}:{out.value.decode('utf-8') if out.value else 'unknown'}"
        except Exception:
            return f"error={code}"
    except Exception:
        return "unavailable"
