"""
Управление воспроизводимостью через random seed.

На MPS используется CPU generator для стабильной генерации.
"""

import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    """Устанавливает seed для всех генераторов случайных чисел.

    Args:
        seed: Значение seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_generator(seed: int | None, device: str) -> torch.Generator | None:
    """Создаёт генератор случайных чисел для воспроизводимости.

    На MPS генератор создаётся на CPU, так как MPS generator нестабилен.

    Args:
        seed: Значение seed. Если None — генератор не создаётся.
        device: Целевое устройство.

    Returns:
        torch.Generator или None.
    """
    if seed is None:
        return None

    # На MPS используем CPU generator для воспроизводимости
    gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator
