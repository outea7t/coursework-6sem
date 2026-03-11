"""
Рунге-Кутта 4-го порядка (RK4) для probability flow ODE.

Классический метод RK4 — один из самых известных и широко используемых
методов решения ODE. Обеспечивает 4-й порядок точности за счёт
4 вычислений правой части на каждом шаге.

Математика:
===========

    Probability flow ODE: dx/dt = F(x, t)

    Классический RK4:
        k1 = F(x_n, t_n)
        k2 = F(x_n + 0.5*h*k1, t_n + 0.5*h)
        k3 = F(x_n + 0.5*h*k2, t_n + 0.5*h)
        k4 = F(x_n + h*k3, t_n + h)

        x_{n+1} = x_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    где h = dt (размер шага).

Вывод весов:
    Веса (1/6, 2/6, 2/6, 1/6) получаются из условий:
    - Совпадение с разложением Тейлора до O(h^4)
    - k1, k4 оцениваются на концах интервала (вес 1/6)
    - k2, k3 оцениваются в середине (вес 2/6 = 1/3)
    - Это формула Симпсона: (f(a) + 4*f(mid) + f(b)) / 6

Ошибка:
    Локальная ошибка: O(h^5)
    Глобальная ошибка: O(h^4)

    Для сравнения:
    - Euler: O(h)
    - Heun: O(h^2)
    - RK4: O(h^4)

    RK4 значительно точнее, но требует 4 вызова модели на шаг.
    При CFG (x2) это 8 forward passes U-Net на шаг!

Применение в диффузии:
    RK4 позволяет использовать очень мало шагов (10-20) при высокой
    точности. Однако стоимость 4 NFE/шаг делает его сопоставимым
    по скорости с 4x Euler при том же total NFE.

    RK4 с 25 шагами (100 NFE) vs Euler с 100 шагами (100 NFE):
    RK4 даёт значительно лучшее качество при одинаковом бюджете NFE.

Ссылки:
    [1] Runge, C. (1895). "Über die numerische Auflösung von
        Differentialgleichungen."
    [2] Kutta, W. (1901). "Beitrag zur näherungweisen Integration
        totaler Differentialgleichungen."
"""

import torch
from torch import Tensor
from typing import Callable

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


class RungeKutta4Solver(BaseSolver):
    """Рунге-Кутта 4-го порядка (классический RK4).

    Требует 4 вызова модели на каждый шаг, но обеспечивает
    4-й порядок точности.

    Args:
        sde: SDE модель.
        num_steps: Количество шагов.
    """

    def __init__(self, sde: BaseSDE, num_steps: int = 25) -> None:
        super().__init__(sde=sde, num_steps=num_steps)
        self._model_fn: Callable | None = None

    def set_model_fn(self, model_fn: Callable) -> None:
        """Устанавливает функцию модели для промежуточных оценок.

        RK4 требует 4 вызова модели на шаг. model_fn должна
        принимать (x, t) и возвращать предсказание шума.

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
        """Один шаг RK4.

        Алгоритм:
            h = t_prev - t  (< 0, идём назад)
            k1 = F(x, t)                           — model_output
            k2 = F(x + 0.5*h*k1, t + 0.5*h)       — промежуточная точка
            k3 = F(x + 0.5*h*k2, t + 0.5*h)       — промежуточная точка
            k4 = F(x + h*k3, t + h)                — конечная точка
            x_prev = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

        При отсутствии model_fn, fallback на Euler с model_output.

        Args:
            x: Текущие латенты.
            t: Текущее время.
            t_prev: Следующее время.
            model_output: Предсказание шума в точке (x, t).

        Returns:
            Латенты после одного шага.
        """
        h = t_prev - t  # < 0
        t_mid = t + 0.5 * h  # промежуточное время

        # k1: ODE drift в текущей точке (используем model_output)
        k1 = self._compute_ode_drift(x, t, model_output)

        if self._model_fn is None:
            # Fallback на Euler при отсутствии model_fn
            return x + k1 * h

        # k2: ODE drift в промежуточной точке
        x_mid1 = x + 0.5 * h * k1
        noise_mid1 = self._model_fn(x_mid1, t_mid)
        k2 = self._compute_ode_drift(x_mid1, t_mid, noise_mid1)

        # k3: ODE drift в другой промежуточной точке
        x_mid2 = x + 0.5 * h * k2
        noise_mid2 = self._model_fn(x_mid2, t_mid)
        k3 = self._compute_ode_drift(x_mid2, t_mid, noise_mid2)

        # k4: ODE drift в конечной точке
        x_end = x + h * k3
        noise_end = self._model_fn(x_end, t_prev)
        k4 = self._compute_ode_drift(x_end, t_prev, noise_end)

        # Взвешенная комбинация (формула Симпсона)
        # x_prev = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x_prev = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return x_prev

    def _compute_ode_drift(
        self, x: Tensor, t: Tensor, noise_pred: Tensor
    ) -> Tensor:
        """Вычисляет ODE drift из noise prediction.

        F(x, t) = f(x,t) - 0.5*g(t)^2*score(x,t)
        score = -noise / sigma(t)

        Args:
            x: Состояние.
            t: Время.
            noise_pred: Предсказание шума.

        Returns:
            ODE drift.
        """
        _, sigma_t = self.sde.marginal_params_at_t(t)
        while sigma_t.dim() < noise_pred.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        score = -noise_pred / sigma_t.clamp(min=1e-8)
        return self.sde.reverse_ode_drift(x, t, score)

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return 4

    @property
    def nfe_per_step(self) -> int:
        return 4
