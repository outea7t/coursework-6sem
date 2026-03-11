"""
Линейное расписание шума (Linear Noise Schedule).

Классическое расписание из оригинальной статьи DDPM (Ho et al., 2020).
beta(t) линейно растёт от beta_min до beta_max.

Математика:
    beta(t) = beta_min + t * (beta_max - beta_min),  t in [0, 1]

    Интеграл beta(t) по [0, t]:
        integral_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2

    alpha_bar(t) = exp(-integral_0^t beta(s) ds)
                 = exp(-beta_min * t - 0.5 * (beta_max - beta_min) * t^2)

Параметры по умолчанию: beta_min = 0.0001, beta_max = 0.02 (оригинальный DDPM).

Недостаток: при больших t шум добавляется слишком быстро, что приводит к
потере информации. Косинусное расписание решает эту проблему.

Ссылки:
    Ho, J., Jain, A., & Abbeel, P. (2020).
    "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
"""

import torch

from .base_scheduler import BaseScheduler


class LinearScheduler(BaseScheduler):
    """Линейное расписание шума.

    beta(t) = beta_min + t * (beta_max - beta_min)

    Args:
        beta_min: Минимальное значение beta (начало процесса).
        beta_max: Максимальное значение beta (конец процесса).
        num_train_timesteps: Количество дискретных шагов.
    """

    def __init__(
        self,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        num_train_timesteps: int = 1000,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(num_train_timesteps)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Линейная функция beta(t) (непрерывная).

        Дискретная: beta_discrete(t) = beta_min + t * (beta_max - beta_min)
        Непрерывная: beta(t) = N * beta_discrete(t)

        Масштабирование на N = num_train_timesteps необходимо, чтобы
        integral_0^1 beta(s) ds совпадал с суммой дискретных бет.

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        return self.num_train_timesteps * (self.beta_min + t * (self.beta_max - self.beta_min))

    # alpha_bar(t) наследуется от BaseScheduler — интерполяция дискретных
    # alphas_cumprod для точного соответствия значениям обучения U-Net.

    def _compute_discrete_betas(self) -> torch.Tensor:
        """Дискретные beta значения: линейная сетка от beta_min до beta_max."""
        return torch.linspace(
            self.beta_min, self.beta_max, self.num_train_timesteps, dtype=torch.float32
        )
