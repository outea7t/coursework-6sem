# управление воспроизводимостью через seed

import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_generator(seed: int | None, device: str) -> torch.Generator | None:
    if seed is None:
        return None

    # на mps используем cpu generator
    gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator
