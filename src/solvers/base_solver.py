"""
Абстрактный базовый класс для численных солверов.

Солверы решают обратное SDE или probability flow ODE для генерации
изображений из шума. Это вторая ключевая часть с дифференциальными
уравнениями после SDE.

Общая структура обратного процесса:
====================================

    Начиная с x_T ~ N(0, I), мы интегрируем обратное уравнение
    от t=T до t=0 (или t=epsilon), получая x_0.

    Для SDE-солверов (стохастические):
        x_{t-dt} = x_t + f_reverse(x_t, t)*dt + g(t)*sqrt(dt)*z

    Для ODE-солверов (детерминированные):
        x_{t-dt} = x_t + f_ode(x_t, t)*dt

    Разные солверы отличаются способом аппроксимации этих интегралов:
    - Euler: порядок 1, простейший
    - Heun: порядок 2, предиктор-корректор
    - RK4: порядок 4, классический
    - DPM-Solver++: специализированный для диффузии

Временная сетка:
    Интегрирование идёт от t_max (шум) к t_min (данные).
    Размер шагов dt может быть:
    - Равномерным по t
    - Равномерным по log-SNR (оптимально для DPM-Solver)
    - Адаптивным (автоматический подбор)
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from ..sde.base_sde import BaseSDE


class BaseSolver(ABC):
    """Абстрактный базовый класс для солверов обратного SDE/ODE.

    Args:
        sde: SDE, определяющее прямой/обратный процесс.
        num_steps: Количество шагов интегрирования.
    """

    def __init__(self, sde: BaseSDE, num_steps: int = 30) -> None:
        self.sde = sde
        self.num_steps = num_steps
        self.timesteps = self._build_timesteps()

    @abstractmethod
    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один шаг интегрирования от t к t_prev.

        Args:
            x: Текущее состояние, (batch, C, H, W).
            t: Текущее время.
            t_prev: Следующее время (t_prev < t, ближе к 0).
            model_output: Выход модели (noise prediction или score).

        Returns:
            x_{t_prev} — состояние после одного шага.
        """
        ...

    def _build_timesteps(self) -> Tensor:
        """Создание временной сетки от t_max до t_min.

        Равномерная сетка с num_steps + 1 точками.
        timesteps[0] = t_max ≈ 1.0
        timesteps[-1] = t_min ≈ 0.001

        Returns:
            Тензор временных шагов (num_steps + 1,).
        """
        return torch.linspace(
            self.sde.t_max, self.sde.t_min, self.num_steps + 1
        )

    @property
    def is_stochastic(self) -> bool:
        """Является ли солвер стохастическим (SDE) или детерминированным (ODE).

        Стохастические солверы дают разные результаты при каждом запуске.
        Детерминированные — одинаковый seed → одинаковый результат.
        """
        return False

    @property
    def order(self) -> int:
        """Порядок точности солвера.

        Порядок k означает, что локальная ошибка ~ O(dt^{k+1}),
        а глобальная ошибка ~ O(dt^k).
        """
        return 1

    @property
    def nfe_per_step(self) -> int:
        """Number of Function Evaluations per step.

        Количество вызовов U-Net на каждый шаг.
        Для Euler: 1, для Heun: 2, для RK4: 4.
        При CFG каждый NFE = 2 forward pass (cond + uncond).
        """
        return 1
