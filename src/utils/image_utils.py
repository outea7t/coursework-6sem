# вспомогательные функции для работы с картинками

import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    # превращение тензора (массива чисел из torch) в обычную картинку.
    # на входе тензор формы (1, 3, высота, ширина) или (3, высота, ширина),
    # значения лежат примерно в [-1, 1] - так выдаёт vae.

    # если есть лишняя ось батча - убираем её
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # значения от -1..1 переводим в 0..1, лишнее обрезаем clamp-ом
    tensor = (tensor.float().clamp(-1, 1) + 1.0) / 2.0
    # переставляем оси: (каналы, высота, ширина) -> (высота, ширина, каналы),
    # умножаем на 255 и приводим к целым числам - формат, понятный pillow
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def postprocess_latents(latents: torch.Tensor) -> torch.Tensor:
    # нормализация латента (скрытого представления) в диапазон [-1, 1].
    # используется в отладочных скриптах для визуализации латента как картинки
    latents = latents.float()
    latents = (latents - latents.min()) / (latents.max() - latents.min()) * 2 - 1
    return latents


def save_image(
    image: Image.Image,
    output_dir: str,
    prompt: str,
    solver: str,
    steps: int,
    seed: int | None = None,
    fmt: str = "png",
) -> str:
    # сохранение готовой картинки в файл.
    # имя файла собираем из даты, обрезанного запроса и параметров -
    # так удобно искать нужную картинку среди множества сгенерированных

    # создаём папку, если её нет (parents=True - вместе со всеми вложенными)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # дата и время в имя файла - чтобы файлы не перезаписывали друг друга
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # очищаем запрос от символов, которые нельзя использовать в имени файла:
    # оставляем только буквы, цифры, пробелы, дефисы и подчёркивания
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
    safe_prompt = safe_prompt[:50].strip().replace(" ", "_")

    # если задано зерно - добавляем его в имя для воспроизводимости
    seed_str = f"_s{seed}" if seed is not None else ""
    filename = f"{timestamp}_{safe_prompt}_{solver}_{steps}steps{seed_str}.{fmt}"

    filepath = os.path.join(output_dir, filename)
    image.save(filepath, format=fmt.upper())
    return filepath
