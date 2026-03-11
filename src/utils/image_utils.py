"""
Утилиты для обработки и сохранения изображений.
"""

import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Конвертирует тензор изображения в PIL Image.

    Args:
        tensor: Тензор формы (1, C, H, W) или (C, H, W) со значениями в [-1, 1].

    Returns:
        PIL Image в формате RGB.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # [-1, 1] -> [0, 1]
    tensor = (tensor.float().clamp(-1, 1) + 1.0) / 2.0
    # (C, H, W) -> (H, W, C) -> numpy -> uint8
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def postprocess_latents(latents: torch.Tensor) -> torch.Tensor:
    """Нормализует латенты в диапазон [-1, 1] для визуализации.

    Args:
        latents: Тензор латентов.

    Returns:
        Нормализованный тензор.
    """
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
    """Сохраняет изображение с информативным именем файла.

    Args:
        image: PIL Image для сохранения.
        output_dir: Директория для сохранения.
        prompt: Текстовый промпт (для имени файла).
        solver: Имя солвера.
        steps: Количество шагов.
        seed: Значение seed.
        fmt: Формат файла.

    Returns:
        Путь к сохранённому файлу.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize prompt for filename
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
    safe_prompt = safe_prompt[:50].strip().replace(" ", "_")

    seed_str = f"_s{seed}" if seed is not None else ""
    filename = f"{timestamp}_{safe_prompt}_{solver}_{steps}steps{seed_str}.{fmt}"

    filepath = os.path.join(output_dir, filename)
    image.save(filepath, format=fmt.upper())
    return filepath
