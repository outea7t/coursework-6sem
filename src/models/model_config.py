# параметры предобученной модели sdxl, которые нужны
# для расчёта размеров скрытого представления и типа чисел

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    # имя модели на huggingface - стандартная sdxl
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "auto"
    dtype: str = "float16"
    # коэффициент масштабирования vae - используется при декодировании
    vae_scaling_factor: float = 0.13025
    # сколько каналов в скрытом представлении (у sdxl - 4)
    latent_channels: int = 4
    # во сколько раз скрытое представление меньше картинки
    # по каждой стороне (у sdxl - в 8 раз)
    downscale_factor: int = 8
    # размер дискретной сетки времени, на которой обучали u-net
    num_train_timesteps: int = 1000

    def get_torch_dtype(self) -> torch.dtype:
        # перевод строки в тип чисел torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float16)

    def get_latent_shape(self, height: int = 1024, width: int = 1024) -> tuple[int, ...]:
        # размер скрытого представления для картинки заданного размера.
        # для 1024x1024 получаем (1, 4, 128, 128) - один батч,
        # 4 канала, сторона в 8 раз меньше
        return (
            1,
            self.latent_channels,
            height // self.downscale_factor,
            width // self.downscale_factor,
        )
