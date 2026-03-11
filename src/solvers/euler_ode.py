"""
Euler ODE солвер для probability flow ODE.

Детерминированный аналог Euler-Maruyama. Решает probability flow ODE
вместо обратного SDE, что даёт детерминированную генерацию.
Эквивалентен DDIM (Denoising Diffusion Implicit Models).

Математика:
===========

    Probability flow ODE:
        dx/dt = f(x, t) - 0.5 * g(t)^2 * score(x, t)

    Дискретизация Эйлера:
        x_{t-dt} = x_t + [f(x_t, t) - 0.5*g(t)^2*score(x_t, t)] * dt

    Для VP-SDE:
        f(x,t) = -0.5*beta(t)*x
        g(t) = sqrt(beta(t))
        g^2 = beta(t)

        dx/dt = -0.5*beta(t)*x - 0.5*beta(t)*score(x,t)
              = -0.5*beta(t) * (x + score(x,t))

    Используя epsilon-prediction:
        score = -eps/sigma(t)
        dx/dt = -0.5*beta(t)*x + 0.5*beta(t)*eps/sigma(t)

Свойства:
    - Порядок: 1
    - Детерминированный: одинаковый seed → одинаковый результат
    - 1 вызов модели на шаг
    - Быстрее сходится, чем EM (меньше шагов для того же качества)
    - Эквивалент DDIM

Ссылки:
    [1] Song, J., et al. (2021). "Denoising Diffusion Implicit Models" (ICLR 2021).
    [2] Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs."
"""

import torch
from torch import Tensor

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


class EulerODESolver(BaseSolver):
    """Euler ODE солвер (детерминированный).

    Дискретизация probability flow ODE методом Эйлера.
    Эквивалентен DDIM.

    Args:
        sde: SDE модель.
        num_steps: Количество шагов.
    """

    def __init__(self, sde: BaseSDE, num_steps: int = 30) -> None:
        super().__init__(sde=sde, num_steps=num_steps)

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один шаг Euler ODE.

        x_{t-dt} = x_t + ode_drift(x_t, t) * dt

        где ode_drift = f(x,t) - 0.5*g(t)^2*score(x,t)

        Args:
            x: Текущие латенты.
            t: Текущее время.
            t_prev: Следующее время.
            model_output: Предсказание шума epsilon.

        Returns:
            Латенты после одного шага.
        """
        dt = t_prev - t  # < 0

        # Конвертация noise -> score
        _, sigma_t = self.sde.marginal_params_at_t(t)
        while sigma_t.dim() < model_output.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        score = -model_output / sigma_t.clamp(min=1e-8)

        # ODE drift: f(x,t) - 0.5*g(t)^2*score
        ode_drift = self.sde.reverse_ode_drift(x, t, score)

        # Euler step
        x_prev = x + ode_drift * dt

        return x_prev

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return 1

    @property
    def nfe_per_step(self) -> int:
        return 1
