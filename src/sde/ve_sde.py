"""
Variance Exploding SDE (VE-SDE).

Используется в NCSN (Noise Conditional Score Networks) и SMLD
(Score Matching with Langevin Dynamics). Называется "Variance Exploding"
потому что дисперсия x_t неограниченно растёт с увеличением t.

Прямое SDE:
===========

    dx = sqrt(d[sigma^2(t)]/dt) * dw                                    (1)

    Коэффициенты:
        f(x, t) = 0                                (нулевой drift)
        g(t) = sqrt(d[sigma^2(t)]/dt)              (diffusion)

    Отличие от VP-SDE: отсутствует drift, т.е. средняя траектория
    не меняется. Шум просто накапливается.

Параметризация sigma(t):
========================

    Стандартная параметризация — геометрическая последовательность:
        sigma(t) = sigma_min * (sigma_max / sigma_min)^t

    Тогда:
        sigma^2(t) = sigma_min^2 * (sigma_max/sigma_min)^{2t}
        d[sigma^2]/dt = 2*ln(sigma_max/sigma_min) * sigma^2(t)
        g(t) = sigma(t) * sqrt(2*ln(sigma_max/sigma_min))

Маргинальное распределение:
===========================

    q(x_t | x_0) = N(x_t; x_0, sigma(t)^2 * I)

    Вывод:
        Так как drift = 0, средняя траектория стационарна: E[x_t|x_0] = x_0
        Дисперсия накапливается: Var[x_t|x_0] = integral_0^t g(s)^2 ds = sigma(t)^2

    Заметим, что mean = x_0 (не затухает!), а дисперсия растёт.
    Поэтому при больших t: Var[x_t] = Var[x_0] + sigma(t)^2, что
    "взрывается" — отсюда название Variance Exploding.

Предельное распределение:
=========================

    p_T ≈ N(0, sigma_max^2 * I)

    При sigma_max >> ||x_0||, шум доминирует над сигналом.

Ссылки:
    [1] Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating
        Gradients of the Data Distribution" (NeurIPS 2019).
    [2] Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs."
"""

from typing import Tuple

import torch
from torch import Tensor

from .base_sde import BaseSDE
from ..schedulers.base_scheduler import BaseScheduler


class VESDE(BaseSDE):
    """Variance Exploding SDE.

    dx = sqrt(d[sigma^2(t)]/dt) * dw

    sigma(t) = sigma_min * (sigma_max / sigma_min)^t

    Args:
        scheduler: Noise scheduler (используется для совместимости, но
                   VE-SDE имеет свою собственную параметризацию sigma(t)).
        sigma_min: Минимальный уровень шума.
        sigma_max: Максимальный уровень шума.
        t_min: Минимальное время.
        t_max: Максимальное время.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        super().__init__(scheduler=scheduler, t_min=t_min, t_max=t_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = torch.log(torch.tensor(sigma_max / sigma_min))

    def sigma(self, t: Tensor) -> Tensor:
        """Функция уровня шума sigma(t).

        sigma(t) = sigma_min * (sigma_max / sigma_min)^t

        Геометрическая интерполяция между sigma_min и sigma_max
        на логарифмической шкале:
            log(sigma(t)) = (1-t)*log(sigma_min) + t*log(sigma_max)

        Args:
            t: Время, t in [0, 1].

        Returns:
            sigma(t).
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Коэффициент сноса VE-SDE: f(x, t) = 0.

        В VE-SDE drift отсутствует — средняя траектория стационарна.

        Args:
            x: Текущее состояние.
            t: Время.

        Returns:
            Нулевой тензор.
        """
        return torch.zeros_like(x)

    def diffusion(self, t: Tensor) -> Tensor:
        """Коэффициент диффузии VE-SDE.

        g(t) = sigma(t) * sqrt(2 * ln(sigma_max / sigma_min))

        Вывод:
            sigma^2(t) = sigma_min^2 * (sigma_max/sigma_min)^{2t}
            d[sigma^2]/dt = 2*ln(sigma_max/sigma_min) * sigma^2(t)
            g(t) = sqrt(d[sigma^2]/dt) = sigma(t) * sqrt(2*ln(sigma_max/sigma_min))

        Args:
            t: Время.

        Returns:
            g(t).
        """
        sigma_t = self.sigma(t)
        return sigma_t * torch.sqrt(2.0 * self.log_ratio)

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Параметры маргинального распределения q(x_t | x_0).

        q(x_t | x_0) = N(x_t; x_0, sigma(t)^2 * I)

        Отличие от VP-SDE: среднее = x_0 (без затухания).

        Args:
            x_0: Исходные данные.
            t: Время.

        Returns:
            (mean, std): mean = x_0, std = sigma(t).
        """
        sigma_t = self.sigma(t)
        while sigma_t.dim() < x_0.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        mean = x_0  # mean не затухает в VE-SDE
        std = sigma_t.expand_as(x_0)
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        """Сэмплирование из p_T = N(0, sigma_max^2 * I).

        Args:
            shape: Форма тензора.
            device: Устройство.

        Returns:
            Сэмпл из N(0, sigma_max^2 * I).
        """
        return torch.randn(shape, device=device) * self.sigma_max

    def noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        """Конвертация предсказания шума в score function.

        score(x, t) = -noise / sigma(t)

        Args:
            noise: Предсказанный шум.
            t: Время.

        Returns:
            Score function.
        """
        sigma_t = self.sigma(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)
