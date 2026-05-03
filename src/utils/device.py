# определение устройства для вычислений
# и подходящего типа чисел (float16 / float32)

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> str:
    # если пользователь явно указал устройство - берём его
    if preference != "auto":
        return preference

    # автовыбор: сначала пробуем видеокарту apple (mps),
    # потом nvidia (cuda), и только если ни того, ни другого нет -
    # центральный процессор (cpu). на cpu генерация будет очень медленной
    if torch.backends.mps.is_available():
        logger.info("Используем видеокарту Apple (MPS)")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("Используем видеокарту NVIDIA (CUDA)")
        return "cuda"
    else:
        logger.info("Используем центральный процессор (CPU)")
        return "cpu"


def get_dtype(device: str) -> torch.dtype:
    # на видеокартах используем float16 - вдвое меньше памяти
    # и заметно быстрее. на cpu float16 медленнее, поэтому float32
    if device == "cpu":
        return torch.float32
    return torch.float16


def randn_tensor(
    shape: tuple,
    device: str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # генерация массива из случайных чисел нормального распределения.
    # это начальный шум, с которого стартует обратный процесс диффузии.

    # на видеокарте apple генератор не работает - сначала создаём шум
    # на процессоре, потом переносим на устройство
    if device == "mps":
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
        return noise.to(device)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
