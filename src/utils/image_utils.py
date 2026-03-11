# утилиты для изображений

import os
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = (tensor.float().clamp(-1, 1) + 1.0) / 2.0
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(array)


def postprocess_latents(latents: torch.Tensor) -> torch.Tensor:
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)
    safe_prompt = safe_prompt[:50].strip().replace(" ", "_")

    seed_str = f"_s{seed}" if seed is not None else ""
    filename = f"{timestamp}_{safe_prompt}_{solver}_{steps}steps{seed_str}.{fmt}"

    filepath = os.path.join(output_dir, filename)
    image.save(filepath, format=fmt.upper())
    return filepath
