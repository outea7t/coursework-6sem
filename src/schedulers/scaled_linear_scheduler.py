"""
Масштабированное линейное расписание шума (Scaled Linear Noise Schedule).

Используется по умолчанию в Stable Diffusion. Отличие от обычного линейного:
линейная интерполяция производится в пространстве sqrt(beta), что обеспечивает
более плавное нарастание шума.

Математика:
    sqrt(beta(t)) = sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min))
    beta(t) = (sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min)))^2

    Обозначим: a = sqrt(beta_min), b = sqrt(beta_max)
    beta(t) = (a + t*(b - a))^2 = a^2 + 2*a*(b-a)*t + (b-a)^2 * t^2

    Интеграл beta(t) по [0, t]:
        integral_0^t beta(s) ds = a^2 * t + a*(b-a)*t^2 + (b-a)^2 * t^3 / 3

    alpha_bar(t) = exp(-integral_0^t beta(s) ds)

Параметры по умолчанию: beta_min = 0.00085, beta_max = 0.012
(значения из Stable Diffusion).

Ссылки:
    Rombach, R., et al. (2022).
    "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022).
"""

import math

import torch

from .base_scheduler import BaseScheduler


class ScaledLinearScheduler(BaseScheduler):
    """Масштабированное линейное расписание шума.

    Линейная интерполяция в пространстве sqrt(beta):
        sqrt(beta(t)) = sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min))

    Args:
        beta_min: Минимальное значение beta.
        beta_max: Максимальное значение beta.
        num_train_timesteps: Количество дискретных шагов.
    """

    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sqrt_beta_min = math.sqrt(beta_min)
        self.sqrt_beta_max = math.sqrt(beta_max)
        super().__init__(num_train_timesteps)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Масштабированная линейная функция beta(t) (непрерывная).

        Дискретная формула: beta_discrete(t) = (sqrt(beta_min) + t*(sqrt(beta_max) - sqrt(beta_min)))^2
        Непрерывная формула: beta(t) = N * beta_discrete(t)

        Масштабирование на N = num_train_timesteps необходимо, чтобы
        integral_0^1 beta(s) ds совпадал с суммой дискретных бет:
        sum_{i=1}^{N} beta_i ≈ N * integral_0^1 beta_discrete(s) ds

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        sqrt_beta = self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min)
        return self.num_train_timesteps * sqrt_beta ** 2

    # alpha_bar(t) наследуется от BaseScheduler — интерполяция дискретных
    # alphas_cumprod для точного соответствия значениям обучения U-Net.

    def _compute_discrete_betas(self) -> torch.Tensor:
        """Дискретные beta: линейная сетка в пространстве sqrt(beta)."""
        sqrt_betas = torch.linspace(
            self.sqrt_beta_min,
            self.sqrt_beta_max,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        return sqrt_betas ** 2
