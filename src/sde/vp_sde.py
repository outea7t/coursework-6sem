"""
Variance Preserving SDE (VP-SDE).

Это основное SDE, лежащее в основе DDPM и Stable Diffusion.
Называется "Variance Preserving" потому что суммарная дисперсия
(сигнал + шум) сохраняется: Var[x_t] = alpha_bar(t)*Var[x_0] + (1-alpha_bar(t)) ≈ 1.

Прямое SDE:
===========

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw                  (1)

    Это SDE типа Орнштейна-Уленбека с переменным коэффициентом.

    Коэффициенты:
        f(x, t) = -0.5 * beta(t) * x    (drift — линейный по x)
        g(t) = sqrt(beta(t))              (diffusion)

Маргинальное распределение:
===========================

    Для линейного SDE (1) маргинальное распределение гауссовское:

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar(t)) * x_0, (1 - alpha_bar(t)) * I)

    Вывод:
        Решение линейного SDE dx = a(t)*x*dt + b(t)*dw:
            x_t = exp(A(t)) * x_0 + integral_0^t exp(A(t)-A(s)) * b(s) dw_s

        где A(t) = integral_0^t a(s) ds.

        Для VP-SDE: a(t) = -0.5*beta(t), b(t) = sqrt(beta(t))

        A(t) = -0.5 * integral_0^t beta(s) ds

        Среднее:
            E[x_t | x_0] = exp(A(t)) * x_0
                         = exp(-0.5 * integral_0^t beta(s) ds) * x_0
                         = sqrt(alpha_bar(t)) * x_0

            так как alpha_bar(t) = exp(-integral_0^t beta(s) ds)

        Дисперсия (по формуле Ито для стохастического интеграла):
            Var[x_t | x_0] = integral_0^t exp(2*(A(t)-A(s))) * b(s)^2 ds
                           = integral_0^t exp(-integral_s^t beta(u) du) * beta(s) ds

        Можно показать, что Var[x_t | x_0] = 1 - alpha_bar(t).

        Доказательство:
            Пусть phi(t) = integral_0^t beta(s) ds, тогда alpha_bar(t) = exp(-phi(t))

            Var = integral_0^t exp(-(phi(t) - phi(s))) * phi'(s) ds
                = exp(-phi(t)) * integral_0^t exp(phi(s)) * phi'(s) ds
                = exp(-phi(t)) * [exp(phi(t)) - exp(phi(0))]
                = 1 - exp(-phi(t))
                = 1 - alpha_bar(t)  ∎

Обратное SDE:
=============

    dx = [-0.5*beta(t)*x - beta(t)*score(x,t)] dt + sqrt(beta(t)) dw̃

    Подставив f = -0.5*beta*x и g = sqrt(beta):
    reverse drift = f - g^2*score = -0.5*beta*x - beta*score

Probability Flow ODE:
=====================

    dx/dt = -0.5*beta(t)*x - 0.5*beta(t)*score(x,t)
          = -0.5*beta(t) * [x + score(x,t)]

Предельное распределение:
=========================

    При t -> T (большое T): q(x_T) ≈ N(0, I)
    Это следует из того, что alpha_bar(T) -> 0 при T -> inf.

Ссылки:
    [1] Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models."
    [2] Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs."
"""

from typing import Tuple

import torch
from torch import Tensor

from .base_sde import BaseSDE
from ..schedulers.base_scheduler import BaseScheduler


class VPSDE(BaseSDE):
    """Variance Preserving SDE.

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw

    Args:
        scheduler: Noise scheduler, определяющий функцию beta(t).
        t_min: Минимальное время (eps для численной стабильности).
        t_max: Максимальное время.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        super().__init__(scheduler=scheduler, t_min=t_min, t_max=t_max)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Коэффициент сноса VP-SDE: f(x, t) = -0.5 * beta(t) * x.

        Линейный drift по x — характерная черта VP-SDE,
        которая обеспечивает гауссовость маргинального распределения.

        Args:
            x: Текущее состояние, (batch, C, H, W).
            t: Время, скаляр или (batch,).

        Returns:
            f(x, t) = -0.5 * beta(t) * x.
        """
        beta_t = self.scheduler.beta(t)
        # Reshape beta_t для broadcasting с x (batch, C, H, W)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Коэффициент диффузии VP-SDE: g(t) = sqrt(beta(t)).

        Определяет интенсивность добавляемого шума на каждом шаге.

        Args:
            t: Время, скаляр или (batch,).

        Returns:
            g(t) = sqrt(beta(t)).
        """
        return torch.sqrt(self.scheduler.beta(t))

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Параметры маргинального распределения q(x_t | x_0).

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar(t)) * x_0, (1 - alpha_bar(t)) * I)

        Это позволяет сэмплировать x_t за один шаг:
            x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * epsilon

        Args:
            x_0: Исходные данные, (batch, C, H, W).
            t: Время, скаляр или (batch,).

        Returns:
            (mean, std): mean = sqrt(alpha_bar(t)) * x_0, std = sqrt(1 - alpha_bar(t)).
        """
        alpha_bar_t = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        # Reshape для broadcasting
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar_t) * x_0
        std = torch.sqrt(1.0 - alpha_bar_t)
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        """Сэмплирование из предельного распределения p_T = N(0, I).

        Для VP-SDE при t = T (достаточно большом):
            alpha_bar(T) -> 0, поэтому q(x_T) ≈ N(0, I)

        Args:
            shape: Форма тензора.
            device: Устройство.

        Returns:
            Сэмпл из N(0, I).
        """
        return torch.randn(shape, device=device)

    def noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        """Конвертация предсказания шума в score function.

        Связь между noise prediction epsilon и score:
            score(x, t) = -epsilon / sigma(t)

        Вывод:
            Из q(x_t | x_0) = N(mean_t, sigma_t^2 I):
            nabla_x log q(x_t | x_0) = -(x_t - mean_t) / sigma_t^2
                                       = -epsilon / sigma_t

            так как x_t = mean_t + sigma_t * epsilon,
            следовательно epsilon = (x_t - mean_t) / sigma_t

        Args:
            noise: Предсказанный шум epsilon, (batch, C, H, W).
            t: Время.

        Returns:
            Score function -epsilon / sigma(t).
        """
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)

    def score_to_noise(self, score: Tensor, t: Tensor) -> Tensor:
        """Конвертация score function в предсказание шума.

        epsilon = -score * sigma(t)

        Args:
            score: Score function, (batch, C, H, W).
            t: Время.

        Returns:
            Предсказание шума epsilon.
        """
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < score.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -score * sigma_t
