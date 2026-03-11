"""
Euler-Maruyama солвер для обратного SDE.

Простейший численный метод для SDE. Является стохастическим аналогом
метода Эйлера для ODE и эквивалентен оригинальному DDPM сэмплированию.

Математика:
===========

    Обратное SDE:
        dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw̃

    Дискретизация Эйлера-Маруямы:
        x_{t-dt} = x_t + [f(x_t,t) - g(t)^2 * score(x_t,t)] * dt + g(t) * sqrt(|dt|) * z

    где z ~ N(0, I), dt < 0 (идём назад по времени).

    Используя epsilon-prediction вместо score:
        score = -eps / sigma(t)
        drift = f(x,t) + g(t)^2 * eps / sigma(t)

    Для VP-SDE с f(x,t) = -0.5*beta(t)*x и g(t) = sqrt(beta(t)):
        x_{t-dt} = x_t + [-0.5*beta*x_t + beta*eps/sigma(t)] * dt + sqrt(beta) * sqrt(|dt|) * z

Свойства:
    - Порядок сходимости: 0.5 (сильная) / 1.0 (слабая)
    - Стохастический: каждый запуск даёт разный результат
    - Простейшая реализация
    - 1 вызов модели на шаг
    - Эквивалент DDPM sampling

Ссылки:
    [1] Kloeden, P., & Platen, E. (1992). "Numerical Solution of SDEs."
    [2] Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models."
"""

import torch
from torch import Tensor

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


class EulerMaruyamaSolver(BaseSolver):
    """Euler-Maruyama солвер (стохастический).

    Дискретизация обратного SDE методом Эйлера-Маруямы.
    Эквивалентен оригинальному DDPM сэмплированию.

    Args:
        sde: SDE модель.
        num_steps: Количество шагов.
    """

    def __init__(self, sde: BaseSDE, num_steps: int = 50) -> None:
        super().__init__(sde=sde, num_steps=num_steps)

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один шаг Euler-Maruyama.

        Алгоритм:
            1. Вычислить score из model_output (noise prediction)
            2. Вычислить reverse drift: f - g^2 * score
            3. Вычислить стохастический член: g * sqrt(|dt|) * z
            4. x_{t-dt} = x_t + drift * dt + diffusion_term

        Args:
            x: Текущие латенты (batch, C, H, W).
            t: Текущее время.
            t_prev: Следующее время (t_prev < t).
            model_output: Предсказание шума epsilon.

        Returns:
            Латенты после одного шага.
        """
        # dt = t_prev - t < 0 (идём назад по времени)
        dt = t_prev - t

        # Конвертация noise -> score: score = -eps / sigma(t)
        _, sigma_t = self.sde.marginal_params_at_t(t)
        while sigma_t.dim() < model_output.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        score = -model_output / sigma_t.clamp(min=1e-8)

        # Reverse drift: f(x,t) - g(t)^2 * score
        drift = self.sde.reverse_drift(x, t, score)

        # Коэффициент диффузии
        g = self.sde.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)

        # Стохастический член: g * sqrt(|dt|) * z
        # Используем |dt| так как dt < 0, а sqrt от отрицательного не определён
        noise = torch.randn_like(x)

        # Для последнего шага (t_prev ≈ t_min) не добавляем шум
        if t_prev < self.sde.t_min + 1e-4:
            stochastic_term = 0.0
        else:
            stochastic_term = g * torch.sqrt(torch.abs(dt)) * noise

        # x_{t-dt} = x_t + drift * dt + stochastic_term
        x_prev = x + drift * dt + stochastic_term

        return x_prev

    @property
    def is_stochastic(self) -> bool:
        return True

    @property
    def order(self) -> int:
        return 1

    @property
    def nfe_per_step(self) -> int:
        return 1
