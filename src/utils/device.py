# определение устройства и типа данных

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference

    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS backend")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("Using CUDA backend")
        return "cuda"
    else:
        logger.info("Using CPU backend")
        return "cpu"


def get_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return torch.float16


def randn_tensor(
    shape: tuple,
    device: str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # на mps генерация на cpu, потом перенос
    if device == "mps":
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return noise.to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
