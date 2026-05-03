# работа с зерном случайности
# seed нужен, чтобы при одних и тех же параметрах генерация
# давала одинаковую картинку. без него каждый запуск будет разным,
# и нельзя будет повторить понравившийся результат

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    # выставляем зерно во все три библиотеки, которые могут
    # генерировать случайные числа: стандартный random, numpy и torch.
    # если хоть одну пропустить - поведение перестаёт быть воспроизводимым
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_generator(seed: int | None, device: str) -> torch.Generator | None:
    # отдельный генератор torch для создания начального шума.
    # если seed не задан - возвращаем None, тогда torch берёт глобальное зерно
    if seed is None:
        return None

    # видеокарта apple (mps) не поддерживает свой генератор -
    # делаем генератор на процессоре, потом перенесём шум на устройство
    if device == "mps":
        gen_device = "cpu"
    else:
        gen_device = device

    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    return generator
