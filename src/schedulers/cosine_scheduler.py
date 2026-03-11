"""
Косинусное расписание шума (Cosine Noise Schedule).

Предложено в статье Nichol & Dhariwal (2021). Обеспечивает более равномерное
распределение уровня шума по временным шагам, что улучшает качество
генерации, особенно на низких разрешениях.

Математика:
    alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2)

    где s = 0.008 — малый сдвиг для предотвращения слишком малых значений
    beta при t -> 0.

    beta(t) вычисляется через производную:
        beta(t) = -d/dt [log(alpha_bar(t))]
                = pi / (1 + s) * tan((t + s) / (1 + s) * pi/2)

    Дискретная версия:
        beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        beta_t = min(beta_t, 0.999)  # clipping для стабильности

Преимущества над линейным расписанием:
    - Меньше шума на ранних шагах → сохраняется больше структуры
    - Более плавный переход между уровнями шума
    - Лучшее качество на низких разрешениях (64x64, 128x128)

Ссылки:
    Nichol, A., & Dhariwal, P. (2021).
    "Improved Denoising Diffusion Probabilistic Models" (ICML 2021).
"""

import math

import torch

from .base_scheduler import BaseScheduler


class CosineScheduler(BaseScheduler):
    """Косинусное расписание шума.

    alpha_bar(t) = cos^2((t + s) / (1 + s) * pi/2)

    Args:
        s: Смещение для предотвращения вырождения при t -> 0.
        num_train_timesteps: Количество дискретных шагов.
    """

    def __init__(
        self,
        s: float = 0.008,
        num_train_timesteps: int = 1000,
    ) -> None:
        self.s = s
        super().__init__(num_train_timesteps)

    # alpha_bar(t) наследуется от BaseScheduler — интерполяция дискретных
    # alphas_cumprod для точного соответствия значениям обучения U-Net.

    def _alpha_bar_cosine(self, t: torch.Tensor) -> torch.Tensor:
        """Аналитическая косинусная alpha_bar(t) для вычисления дискретных бет.

        alpha_bar(t) = cos^2((t + s) / (1 + s) * pi/2)

        Используется только в _compute_discrete_betas для инициализации.
        """
        angle = (t + self.s) / (1.0 + self.s) * (math.pi / 2.0)
        f_t = torch.cos(angle) ** 2
        angle_0 = torch.tensor(self.s / (1.0 + self.s) * (math.pi / 2.0))
        f_0 = torch.cos(angle_0) ** 2
        return f_t / f_0

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Мгновенное значение beta(t), вычисленное через производную alpha_bar(t).

        beta(t) = -d/dt [log(alpha_bar(t))]

        Вывод:
            d/dt alpha_bar(t) = -pi/(1+s) * cos(theta) * sin(theta)
                              = -pi/(2(1+s)) * sin(2*theta)
            где theta = (t+s)/(1+s) * pi/2

            beta(t) = -(1/alpha_bar) * d/dt alpha_bar
                    = pi/(1+s) * tan(theta)

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        angle = (t + self.s) / (1.0 + self.s) * (math.pi / 2.0)
        return (math.pi / (1.0 + self.s)) * torch.tan(angle)

    def _compute_discrete_betas(self) -> torch.Tensor:
        """Дискретные beta из cosine alpha_bar.

        beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, с clipping до 0.999.
        Использует аналитическую формулу (_alpha_bar_cosine) для избежания
        циклической зависимости с BaseScheduler.alpha_bar.
        """
        steps = self.num_train_timesteps
        t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        alpha_bars = self._alpha_bar_cosine(t)

        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        return betas.clamp(max=0.999).float()
