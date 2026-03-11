"""
Sub-VP SDE (Sub-Variance Preserving SDE).

Вариант VP-SDE, в котором дисперсия маргинального распределения
всегда строго меньше 1 - alpha_bar(t). Это обеспечивает лучшее
предельное распределение при конечном T.

Прямое SDE:
===========

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t) * (1 - exp(-2*B(t)))) * dw

    где B(t) = integral_0^t beta(s) ds — кумулятивный интеграл beta.

    Коэффициенты:
        f(x, t) = -0.5 * beta(t) * x    (тот же drift, что и VP-SDE)
        g(t) = sqrt(beta(t) * (1 - exp(-2*B(t))))  (модифицированная diffusion)

Отличие от VP-SDE:
==================

    В VP-SDE: g(t) = sqrt(beta(t))
    В Sub-VP SDE: g(t) = sqrt(beta(t) * (1 - exp(-2*B(t))))

    Множитель (1 - exp(-2*B(t))) < 1, поэтому шум добавляется
    медленнее, чем в VP-SDE.

Маргинальное распределение:
===========================

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar(t)) * x_0, [1 - exp(-2*B(t))]*alpha_bar(t) * I)

    Обратите внимание: дисперсия = [1 - exp(-2*B(t))] * alpha_bar(t)
    В VP-SDE: дисперсия = 1 - alpha_bar(t)

    При t -> 0: обе формулы дают 0
    При t -> inf: Sub-VP дисперсия -> alpha_bar(t) -> 0,
                  в то время как VP дисперсия -> 1

    Следствие: предельное распределение Sub-VP SDE — дельта-функция
    в нуле, а не N(0, I). Это означает, что Sub-VP SDE лучше
    аппроксимирует нулевое предельное состояние.

Ссылки:
    Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs."
    Appendix C.
"""

import math
from typing import Tuple

import torch
from torch import Tensor

from .base_sde import BaseSDE
from ..schedulers.base_scheduler import BaseScheduler


class SubVPSDE(BaseSDE):
    """Sub-Variance Preserving SDE.

    dx = -0.5*beta(t)*x*dt + sqrt(beta(t)*(1-exp(-2*B(t))))*dw

    Args:
        scheduler: Noise scheduler.
        t_min: Минимальное время.
        t_max: Максимальное время.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        super().__init__(scheduler=scheduler, t_min=t_min, t_max=t_max)

    def _integrated_beta(self, t: Tensor) -> Tensor:
        """Кумулятивный интеграл beta: B(t) = integral_0^t beta(s) ds.

        Вычисляется из alpha_bar:
            alpha_bar(t) = exp(-B(t))  [для VP-SDE с 0.5 множителем]
            или alpha_bar(t) = exp(-B(t))

        Здесь используем: B(t) = -log(alpha_bar(t))

        Args:
            t: Время.

        Returns:
            B(t) = -log(alpha_bar(t)).
        """
        ab = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        return -torch.log(ab)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Коэффициент сноса: f(x, t) = -0.5 * beta(t) * x.

        Тот же drift, что и в VP-SDE.

        Args:
            x: Текущее состояние.
            t: Время.

        Returns:
            f(x, t).
        """
        beta_t = self.scheduler.beta(t)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Коэффициент диффузии Sub-VP SDE.

        g(t) = sqrt(beta(t) * (1 - exp(-2*B(t))))

        Множитель (1 - exp(-2*B(t))) обеспечивает "суб-вариантность":
        - При t -> 0: B(t) -> 0, множитель -> 0, шум минимален
        - При t -> inf: B(t) -> inf, множитель -> 1, сходится к VP-SDE

        Args:
            t: Время.

        Returns:
            g(t).
        """
        beta_t = self.scheduler.beta(t)
        B_t = self._integrated_beta(t)
        discount = 1.0 - torch.exp(-2.0 * B_t)
        return torch.sqrt(beta_t * discount.clamp(min=1e-8))

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Параметры маргинального распределения q(x_t | x_0).

        Среднее: sqrt(alpha_bar(t)) * x_0
        Дисперсия: 1 - exp(-2*B(t)) * alpha_bar(t)
            = 1 - alpha_bar(t)^2  (для VP-SDE с B(t) = -log(alpha_bar(t)))

        Стандартное отклонение: sqrt(1 - alpha_bar(t)^2)

        Сравнение:
            VP-SDE:     std = sqrt(1 - alpha_bar(t))
            Sub-VP SDE: std = sqrt(1 - alpha_bar(t)^2) <= sqrt(1 - alpha_bar(t))

            Неравенство следует из alpha_bar(t) in (0, 1]:
            alpha_bar^2 >= alpha_bar, поэтому 1 - alpha_bar^2 <= 1 - alpha_bar

        Args:
            x_0: Исходные данные.
            t: Время.

        Returns:
            (mean, std).
        """
        alpha_bar_t = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar_t) * x_0
        # std = sqrt(1 - alpha_bar^2)
        std = torch.sqrt((1.0 - alpha_bar_t ** 2).clamp(min=1e-8))
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        """Сэмплирование из предельного распределения.

        Предельное распределение Sub-VP SDE — дельта-функция в 0.
        На практике используем N(0, I) как и в VP-SDE, так как
        при конечном T отличие минимально.

        Args:
            shape: Форма тензора.
            device: Устройство.

        Returns:
            Сэмпл из N(0, I).
        """
        return torch.randn(shape, device=device)
