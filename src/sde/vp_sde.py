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

from ..schedulers.scaled_linear_scheduler import ScaledLinearScheduler


class VPSDE:
    """Variance Preserving SDE.

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw

    Args:
        scheduler: Noise scheduler, определяющий функцию beta(t).
        t_min: Минимальное время (eps для численной стабильности).
        t_max: Максимальное время.
    """

    def __init__(
        self,
        scheduler: ScaledLinearScheduler,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.scheduler = scheduler
        self.t_min = t_min
        self.t_max = t_max

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Коэффициент сноса VP-SDE: f(x, t) = -0.5 * beta(t) * x.

        Args:
            x: Текущее состояние, (batch, C, H, W).
            t: Время, скаляр или (batch,).

        Returns:
            f(x, t) = -0.5 * beta(t) * x.
        """
        beta_t = self.scheduler.beta(t)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Коэффициент диффузии VP-SDE: g(t) = sqrt(beta(t)).

        Args:
            t: Время, скаляр или (batch,).

        Returns:
            g(t) = sqrt(beta(t)).
        """
        return torch.sqrt(self.scheduler.beta(t))

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Параметры маргинального распределения q(x_t | x_0).

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar(t)) * x_0, (1 - alpha_bar(t)) * I)

        Args:
            x_0: Исходные данные, (batch, C, H, W).
            t: Время, скаляр или (batch,).

        Returns:
            (mean, std): mean = sqrt(alpha_bar(t)) * x_0, std = sqrt(1 - alpha_bar(t)).
        """
        alpha_bar_t = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar_t) * x_0
        std = torch.sqrt(1.0 - alpha_bar_t)
        return mean, std

    def marginal_params_at_t(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Коэффициенты маргинального распределения (без x_0).

        Возвращает (sqrt(alpha_bar(t)), sqrt(1 - alpha_bar(t))),
        т.е. коэффициенты в формуле x_t = mean_coeff * x_0 + std * eps.

        Args:
            t: Время.

        Returns:
            (mean_coeff, std): коэффициенты маргинала.
        """
        ab = self.scheduler.alpha_bar(t)
        ab = ab.clamp(min=1e-8)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1.0 - ab)
        return mean_coeff, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        """Сэмплирование из предельного распределения p_T = N(0, I).

        Args:
            shape: Форма тензора.
            device: Устройство.

        Returns:
            Сэмпл из N(0, I).
        """
        return torch.randn(shape, device=device)

    def reverse_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        """Drift обратного SDE (уравнение Андерсона).

        f_reverse(x, t) = f(x, t) - g(t)^2 * score(x, t)

        Args:
            x: Текущее состояние.
            t: Текущее время.
            score: Оценка score function nabla_x log p_t(x).

        Returns:
            Drift обратного SDE.
        """
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - g ** 2 * score

    def reverse_ode_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        """Drift probability flow ODE.

        f_ode(x, t) = f(x, t) - 0.5 * g(t)^2 * score(x, t)

        Args:
            x: Текущее состояние.
            t: Текущее время.
            score: Оценка score function.

        Returns:
            Drift probability flow ODE.
        """
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - 0.5 * g ** 2 * score

    def noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        """Конвертация предсказания шума в score function.

        score(x, t) = -epsilon / sigma(t)

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
