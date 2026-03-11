"""
Метод Хойна (Heun's method) — предиктор-корректор для probability flow ODE.

Метод Хойна — метод второго порядка, использующий стратегию
"предиктор-корректор": сначала делает предварительный шаг Эйлера
(предиктор), затем уточняет результат (корректор).

Математика:
===========

    Probability flow ODE: dx/dt = F(x, t)
    где F(x, t) = f(x,t) - 0.5*g(t)^2*score(x,t)

    Шаг Хойна:
        1. Предиктор (Euler):
           x̃ = x_t + F(x_t, t) * dt

        2. Корректор (трапецоидальное правило):
           x_{t-dt} = x_t + 0.5 * [F(x_t, t) + F(x̃, t_prev)] * dt

    Это метод трапеций, использующий оценку производной в двух точках:
    начальной (x_t, t) и конечной (x̃, t_prev).

Вывод ошибки:
    Метод Эйлера (порядок 1): x_{n+1} = x_n + h*f(x_n)
        Ошибка: O(h^2) на шаг, O(h) глобальная

    Метод Хойна (порядок 2): x_{n+1} = x_n + h/2*(f(x_n) + f(x̃_n))
        Ошибка: O(h^3) на шаг, O(h^2) глобальная

    Разложение Тейлора показывает, что Хойн точнее Эйлера
    в (h/2) раз, что существенно при малых h.

Свойства:
    - Порядок: 2
    - Детерминированный
    - 2 вызова модели (NFE) на шаг
    - Хороший баланс между качеством и скоростью
    - Используется в karras samplers (Elucidating Diffusion)

Ссылки:
    [1] Heun, K. (1900). "Neue Methoden zur approximativen Integration."
    [2] Karras, T., et al. (2022). "Elucidating the Design Space of
        Diffusion-Based Generative Models" (NeurIPS 2022).
"""

import torch
from torch import Tensor
from typing import Callable

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


class HeunSolver(BaseSolver):
    """Метод Хойна (предиктор-корректор, порядок 2).

    Args:
        sde: SDE модель.
        num_steps: Количество шагов.
    """

    def __init__(self, sde: BaseSDE, num_steps: int = 30) -> None:
        super().__init__(sde=sde, num_steps=num_steps)
        self._model_fn: Callable | None = None

    def set_model_fn(self, model_fn: Callable) -> None:
        """Устанавливает функцию модели для промежуточных оценок.

        Для метода Хойна нужен второй вызов модели на промежуточной точке.
        Эта функция должна принимать (x, t) и возвращать предсказание шума.

        Args:
            model_fn: Callable(x, t) -> noise_prediction.
        """
        self._model_fn = model_fn

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один шаг метода Хойна.

        Алгоритм:
            1. F1 = ode_drift(x_t, t)          (используя model_output)
            2. x̃ = x_t + F1 * dt              (предиктор Эйлера)
            3. F2 = ode_drift(x̃, t_prev)       (второй вызов модели)
            4. x_{t-dt} = x_t + 0.5*(F1 + F2)*dt (корректор)

        При отсутствии model_fn для второго вызова, fallback на Euler.

        Args:
            x: Текущие латенты.
            t: Текущее время.
            t_prev: Следующее время.
            model_output: Предсказание шума в точке (x, t).

        Returns:
            Латенты после одного шага.
        """
        dt = t_prev - t  # < 0

        # Шаг 1: ODE drift в текущей точке
        score_1 = self._noise_to_score(model_output, t)
        drift_1 = self.sde.reverse_ode_drift(x, t, score_1)

        # Шаг 2: Предиктор (Euler)
        x_pred = x + drift_1 * dt

        # Шаг 3: Если есть model_fn — вычисляем drift в предсказанной точке
        if self._model_fn is not None and t_prev > self.sde.t_min + 1e-4:
            # Второй вызов модели
            noise_pred_2 = self._model_fn(x_pred, t_prev)
            score_2 = self._noise_to_score(noise_pred_2, t_prev)
            drift_2 = self.sde.reverse_ode_drift(x_pred, t_prev, score_2)

            # Шаг 4: Корректор (трапецоидальное правило)
            # x_{t-dt} = x_t + 0.5*(F1 + F2)*dt
            x_prev = x + 0.5 * (drift_1 + drift_2) * dt
        else:
            # Fallback на Euler (для последнего шага или без model_fn)
            x_prev = x_pred

        return x_prev

    def _noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        """Конвертация noise prediction в score."""
        _, sigma_t = self.sde.marginal_params_at_t(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return 2

    @property
    def nfe_per_step(self) -> int:
        return 2
