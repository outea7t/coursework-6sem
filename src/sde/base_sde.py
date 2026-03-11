"""
Абстрактный базовый класс для стохастических дифференциальных уравнений (SDE).

Теоретическое обоснование:
=========================

Стохастическое дифференциальное уравнение (SDE) описывает эволюцию
случайного процесса x(t) во времени:

    dx = f(x, t) dt + g(t) dw                                          (1)

где:
    - f(x, t) — коэффициент сноса (drift), определяет детерминированную
      компоненту эволюции
    - g(t) — коэффициент диффузии (diffusion), определяет интенсивность
      случайного шума
    - dw — приращение Винеровского процесса (Brownian motion)

Прямой процесс (Forward SDE):
    Начиная с данных x_0 ~ p_data, прямое SDE (1) постепенно добавляет
    шум, превращая x_0 в чистый шум x_T ~ N(0, I).

Обратный процесс (Reverse SDE):
    Ключевой результат Андерсона (1982): обратное по времени SDE имеет вид:

    dx = [f(x, t) - g(t)^2 * nabla_x log p_t(x)] dt + g(t) dw̃       (2)

    где dw̃ — обратный Винеровский процесс, а nabla_x log p_t(x) —
    score function (градиент логарифма плотности).

Probability Flow ODE:
    Существует детерминированное ODE, маргинальные распределения которого
    совпадают с SDE (2):

    dx = [f(x, t) - 0.5 * g(t)^2 * nabla_x log p_t(x)] dt            (3)

    Это ODE позволяет:
    - Детерминированную генерацию (одинаковый seed → одинаковый результат)
    - Точное вычисление log-likelihood
    - Использование ODE солверов (RK4, DPM-Solver++)

Связь с нейронной сетью:
    Score function аппроксимируется нейронной сетью s_theta(x, t):
    - При epsilon-prediction: s_theta(x, t) = -epsilon_theta(x, t) / sigma(t)
    - При score-prediction: s_theta(x, t) напрямую

Ссылки:
    [1] Anderson, B. (1982). "Reverse-time diffusion equation models."
    [2] Song, Y., et al. (2021). "Score-Based Generative Modeling through
        Stochastic Differential Equations" (ICLR 2021).
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ..schedulers.base_scheduler import BaseScheduler


class BaseSDE(ABC):
    """Абстрактный базовый класс для SDE.

    Определяет интерфейс для прямого и обратного процессов.
    Каждая конкретная реализация (VP-SDE, VE-SDE, Sub-VP SDE)
    специфицирует конкретные формы f(x,t) и g(t).

    Args:
        scheduler: Noise scheduler, определяющий beta(t).
        t_min: Минимальное время (epsilon > 0 для численной стабильности).
        t_max: Максимальное время.
    """

    def __init__(
        self,
        scheduler: BaseScheduler,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.scheduler = scheduler
        self.t_min = t_min
        self.t_max = t_max

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Коэффициент сноса прямого SDE: f(x, t).

        В уравнении dx = f(x, t) dt + g(t) dw
        drift определяет детерминированную часть эволюции.

        Args:
            x: Текущее состояние, форма (batch, ...).
            t: Текущее время, форма (batch,) или скаляр.

        Returns:
            Drift f(x, t), форма совпадает с x.
        """
        ...

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """Коэффициент диффузии прямого SDE: g(t).

        В уравнении dx = f(x, t) dt + g(t) dw
        diffusion определяет интенсивность стохастического шума.

        Args:
            t: Текущее время, форма (batch,) или скаляр.

        Returns:
            Diffusion g(t), скаляр или форма (batch,).
        """
        ...

    @abstractmethod
    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Параметры маргинального распределения q(x_t | x_0).

        Для SDE с линейным drift (VP, VE) маргинальное распределение
        является гауссовским:
            q(x_t | x_0) = N(x_t; mean(t)*x_0, std(t)^2 * I)

        Это позволяет сэмплировать x_t напрямую без пошагового
        интегрирования прямого SDE:
            x_t = mean(t) * x_0 + std(t) * epsilon,  epsilon ~ N(0, I)

        Args:
            x_0: Начальные данные, форма (batch, ...).
            t: Время, форма (batch,) или скаляр.

        Returns:
            (mean, std): Среднее и стандартное отклонение q(x_t | x_0).
        """
        ...

    @abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        """Сэмплирование из предельного (prior) распределения p_T.

        При t -> T прямой процесс должен приводить к известному
        простому распределению (обычно N(0, I)).

        Args:
            shape: Форма выходного тензора.
            device: Устройство для тензора.

        Returns:
            Тензор-сэмпл из p_T, форма shape.
        """
        ...

    def reverse_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        """Drift обратного SDE (уравнение Андерсона).

        Обратное SDE:
            dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dw̃

        Reverse drift:
            f_reverse(x, t) = f(x, t) - g(t)^2 * score(x, t)

        Вывод:
            Из уравнения Андерсона (1982), обратное по времени SDE
            для процесса dx = f*dt + g*dw имеет drift:
            f̃(x, t) = f(x, t) - g(t)^2 * nabla_x log p_t(x)

            Знак минус перед g^2*score появляется потому что обратный
            процесс "отменяет" добавление шума.

        Args:
            x: Текущее состояние.
            t: Текущее время.
            score: Оценка score function nabla_x log p_t(x).

        Returns:
            Drift обратного SDE.
        """
        f = self.drift(x, t)
        g = self.diffusion(t)
        # Reshape g для broadcasting с x
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - g ** 2 * score

    def reverse_ode_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        """Drift probability flow ODE.

        Probability flow ODE:
            dx = [f(x,t) - 0.5 * g(t)^2 * score(x,t)] dt

        Это детерминированное ODE, маргинальные распределения которого
        совпадают с обратным SDE. Множитель 0.5 перед g^2 (вместо 1.0
        в обратном SDE) компенсирует отсутствие стохастического члена.

        Доказательство:
            Уравнение Фоккера-Планка для SDE с drift f и diffusion g:
            dp/dt = -div(f*p) + 0.5*g^2*Laplacian(p)

            Для ODE с drift f_ode:
            dp/dt = -div(f_ode*p)

            Приравнивая: f_ode = f - 0.5*g^2*nabla log p

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
        # Ensure we don't take sqrt of negative
        ab = ab.clamp(min=1e-8)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1.0 - ab)
        return mean_coeff, std
