"""
Адаптивный солвер RK45 (Dormand-Prince) для probability flow ODE.

Адаптивный солвер автоматически подбирает размер шага на основе
оценки локальной ошибки. Использует вложенные методы Рунге-Кутта
5-го и 4-го порядка (метод Дорманда-Принса).

Ключевая идея:
==============

    Вместо фиксированной сетки по t, адаптивный солвер:
    1. Делает шаг с помощью RK5 (5-й порядок) — "точное" решение
    2. Делает шаг с помощью RK4 (4-й порядок) — "приближённое" решение
    3. Сравнивает разницу — оценка ошибки
    4. Если ошибка > tolerance: уменьшить шаг и повторить
    5. Если ошибка < tolerance: принять шаг, увеличить следующий шаг

    Оба решения (RK4 и RK5) вычисляются из тех же 6 оценок F(x, t),
    что делает метод эффективным.

Коэффициенты Дорманда-Принса:
    Таблица Бутчера — набор коэффициентов для вычисления k1, ..., k7.
    Эти коэффициенты подобраны для минимизации ошибки усечения.

Управление шагом:
    Новый размер шага:
        h_new = h * min(max_factor, max(min_factor, safety * (tol/err)^(1/5)))

    где safety ≈ 0.9, min_factor ≈ 0.2, max_factor ≈ 5.0

Преимущества:
    - Автоматический подбор шага — не нужно угадывать num_steps
    - Больше шагов на "сложных" участках (начало денойзинга)
    - Меньше шагов на "простых" участках (конец)
    - Гарантия точности через контроль ошибки

Недостатки:
    - Непредсказуемое количество NFE (зависит от tolerance)
    - 6 NFE на шаг (при CFG = 12 forward passes!)
    - Сложнее реализация

Ссылки:
    [1] Dormand, J., & Prince, P. (1980). "A family of embedded Runge-Kutta formulae."
    [2] Hairer, E., et al. (1993). "Solving Ordinary Differential Equations I."
"""

import torch
from torch import Tensor
from typing import Callable

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


# Коэффициенты Дорманда-Принса (DOPRI5)
# Таблица Бутчера для вложенного RK метода 5(4)-го порядка
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]

# Узлы (c_i)
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]

# Веса 5-го порядка (b_i)
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]

# Веса 4-го порядка (b*_i) для оценки ошибки
_B4 = [
    5179 / 57600, 0, 7571 / 16695, 393 / 640,
    -92097 / 339200, 187 / 2100, 1 / 40,
]

# Разность весов для оценки ошибки: E_i = B5_i - B4_i
_E = [b5 - b4 for b5, b4 in zip(_B5, _B4)]


class AdaptiveRK45Solver(BaseSolver):
    """Адаптивный RK45 (Dormand-Prince) солвер.

    Автоматически подбирает размер шага для контроля ошибки.

    Args:
        sde: SDE модель.
        num_steps: Начальное (максимальное) количество шагов.
        atol: Абсолютная точность.
        rtol: Относительная точность.
        max_steps: Максимальное количество шагов (защита от зацикливания).
        safety: Коэффициент безопасности для адаптации шага.
    """

    def __init__(
        self,
        sde: BaseSDE,
        num_steps: int = 50,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        max_steps: int = 200,
        safety: float = 0.9,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety = safety
        self.min_factor = 0.2
        self.max_factor = 5.0
        super().__init__(sde=sde, num_steps=num_steps)
        self._model_fn: Callable | None = None
        self._total_nfe = 0

    def set_model_fn(self, model_fn: Callable) -> None:
        """Устанавливает функцию модели."""
        self._model_fn = model_fn

    def reset(self) -> None:
        """Сбрасывает счётчик NFE."""
        self._total_nfe = 0

    @property
    def total_nfe(self) -> int:
        """Общее количество вызовов модели."""
        return self._total_nfe

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один адаптивный шаг RK45.

        Если model_fn установлен, выполняет полный адаптивный шаг.
        Иначе — fallback на простой Euler.

        Args:
            x: Текущие латенты.
            t: Текущее время.
            t_prev: Целевое время.
            model_output: Предсказание шума (для начальной оценки).

        Returns:
            Латенты после шага.
        """
        if self._model_fn is None:
            # Fallback: Euler step
            dt = t_prev - t
            score = self._noise_to_score(model_output, t)
            drift = self.sde.reverse_ode_drift(x, t, score)
            return x + drift * dt

        return self._adaptive_step(x, t, t_prev)

    def _adaptive_step(
        self, x: Tensor, t_start: Tensor, t_end: Tensor
    ) -> Tensor:
        """Адаптивное интегрирование от t_start до t_end.

        Может выполнить несколько подшагов с автоматическим
        подбором размера шага.

        Args:
            x: Начальное состояние.
            t_start: Начальное время.
            t_end: Конечное время.

        Returns:
            Состояние в момент t_end.
        """
        t_cur = t_start.clone()
        x_cur = x.clone()
        h = t_end - t_start  # Начальный шаг (< 0)

        steps = 0
        while steps < self.max_steps:
            # Не выходить за пределы t_end
            if (h < 0 and t_cur + h < t_end) or (h > 0 and t_cur + h > t_end):
                h = t_end - t_cur

            # Проверка завершения
            if abs(float(t_cur - t_end)) < 1e-8:
                break

            # Попытка шага
            x_new, x_err, nfe = self._dopri5_step(x_cur, t_cur, h)
            self._total_nfe += nfe

            # Оценка ошибки
            error_ratio = self._error_ratio(x_err, x_cur, x_new)

            if error_ratio <= 1.0:
                # Шаг принят
                x_cur = x_new
                t_cur = t_cur + h
                steps += 1

                # Увеличиваем шаг
                factor = self.safety * (1.0 / max(error_ratio, 1e-8)) ** 0.2
                factor = max(self.min_factor, min(self.max_factor, factor))
                h = h * factor
            else:
                # Шаг отвергнут — уменьшаем шаг
                factor = self.safety * (1.0 / max(error_ratio, 1e-8)) ** 0.25
                factor = max(self.min_factor, factor)
                h = h * factor

        return x_cur

    def _dopri5_step(
        self, x: Tensor, t: Tensor, h: Tensor
    ) -> tuple[Tensor, Tensor, int]:
        """Один шаг Dormand-Prince 5(4).

        Вычисляет 7 стадий (k1-k7) и два решения:
        - 5-го порядка (точное)
        - 4-го порядка (для оценки ошибки)

        Args:
            x: Текущее состояние.
            t: Текущее время.
            h: Размер шага.

        Returns:
            (x_new, error, nfe): решение 5-го порядка, оценка ошибки, NFE.
        """
        k = [None] * 7

        # k1
        k[0] = self._f(x, t)

        # k2-k6
        for i in range(1, 6):
            t_i = t + _C[i] * h
            x_i = x + h * sum(
                _A[i][j] * k[j] for j in range(i) if _A[i][j] != 0
            )
            k[i] = self._f(x_i, t_i)

        # k7 (используется в FSAL — First Same As Last)
        t_6 = t + h
        x_6 = x + h * sum(
            _A[6][j] * k[j] for j in range(6) if _A[6][j] != 0
        )
        k[6] = self._f(x_6, t_6)

        # Решение 5-го порядка
        x_new = x + h * sum(_B5[i] * k[i] for i in range(7) if _B5[i] != 0)

        # Оценка ошибки (разность между 5-м и 4-м порядком)
        x_err = h * sum(_E[i] * k[i] for i in range(7) if _E[i] != 0)

        return x_new, x_err, 7  # 7 evaluations

    def _f(self, x: Tensor, t: Tensor) -> Tensor:
        """Правая часть ODE: F(x, t).

        Args:
            x: Состояние.
            t: Время.

        Returns:
            ODE drift.
        """
        noise_pred = self._model_fn(x, t)
        score = self._noise_to_score(noise_pred, t)
        return self.sde.reverse_ode_drift(x, t, score)

    def _noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        """Конвертация noise -> score."""
        _, sigma_t = self.sde.marginal_params_at_t(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)

    def _error_ratio(
        self, err: Tensor, x_old: Tensor, x_new: Tensor
    ) -> float:
        """Вычисляет отношение ошибки к допуску.

        tolerance = atol + rtol * max(|x_old|, |x_new|)
        error_ratio = ||err / tolerance||_RMS

        Args:
            err: Оценка ошибки.
            x_old: Предыдущее состояние.
            x_new: Новое состояние.

        Returns:
            Отношение ошибки к допуску (< 1 = шаг принят).
        """
        tol = self.atol + self.rtol * torch.max(x_old.abs(), x_new.abs())
        ratio = (err / tol.clamp(min=1e-8)).float()
        return float(ratio.pow(2).mean().sqrt())

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return 5

    @property
    def nfe_per_step(self) -> int:
        return 7  # Dormand-Prince uses 7 stages (but FSAL reduces to 6 effective)
