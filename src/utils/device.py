"""
Определение вычислительного устройства и типа данных.

Приоритет: MPS (Apple Silicon) -> CUDA (NVIDIA) -> CPU.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    """Определяет оптимальное вычислительное устройство.

    Args:
        preference: "auto" для автоопределения, или конкретное устройство
                    ("mps", "cuda", "cpu").

    Returns:
        Строка с именем устройства для torch.
    """
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
    """Определяет оптимальный тип данных для устройства.

    На CPU используется float32 для совместимости.
    На MPS и CUDA — float16 для экономии памяти и ускорения.

    Args:
        device: Имя устройства ("mps", "cuda", "cpu").

    Returns:
        torch.dtype — тип данных.
    """
    if device == "cpu":
        return torch.float32
    return torch.float16


def randn_tensor(
    shape: tuple,
    device: str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Безопасная генерация случайного тензора.

    На MPS генерация происходит на CPU с последующим переносом на MPS,
    так как MPS generator может быть нестабилен.

    Args:
        shape: Форма выходного тензора.
        device: Целевое устройство.
        dtype: Тип данных.
        generator: Генератор для воспроизводимости.

    Returns:
        Случайный тензор из N(0, I).
    """
    if device == "mps":
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return noise.to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
