"""
Численная верификация уравнения Фоккера-Планка.

Уравнение Фоккера-Планка (ФП) описывает эволюцию плотности вероятности
p(x, t) для процесса, определяемого SDE. Этот модуль позволяет
убедиться, что наши SDE корректно описывают диффузионный процесс.

Теория:
=======

Для SDE: dx = f(x, t) dt + g(t) dw

Уравнение Фоккера-Планка:
    dp/dt = -d/dx [f(x,t) * p(x,t)] + 0.5 * g(t)^2 * d^2/dx^2 [p(x,t)]

    Первый член: конвективный (перенос плотности drift-ом)
    Второй член: диффузионный (размытие плотности)

Для VP-SDE: f(x,t) = -0.5*beta(t)*x
    dp/dt = 0.5*beta(t) * d/dx [x*p] + 0.5*beta(t) * d^2p/dx^2
          = 0.5*beta(t) * [p + x*dp/dx + d^2p/dx^2]

Аналитическое решение:
    Если p(x, 0) = delta(x - x_0), то:
    p(x, t) = N(x; sqrt(alpha_bar(t))*x_0, (1-alpha_bar(t)))

    Это гауссиана с параметрами, совпадающими с маргинальным
    распределением VP-SDE — что и верифицируем.

Метод верификации:
    1. Решаем уравнение ФП численно (метод конечных разностей)
    2. Сравниваем с аналитическим маргинальным распределением VP-SDE
    3. Вычисляем ошибку (L2 норма разности)

Ссылки:
    [1] Risken, H. (1996). "The Fokker-Planck Equation."
    [2] Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs."
"""

import torch
from torch import Tensor

from ..schedulers.scaled_linear_scheduler import ScaledLinearScheduler


def solve_fokker_planck_1d(
    scheduler: ScaledLinearScheduler,
    x_0: float = 0.0,
    x_range: tuple[float, float] = (-5.0, 5.0),
    nx: int = 500,
    nt: int = 1000,
    t_max: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Численное решение уравнения Фоккера-Планка для VP-SDE в 1D.

    Метод: явная схема конечных разностей (Euler forward).

    dp/dt = 0.5*beta(t) * d/dx[x*p] + 0.5*beta(t) * d^2p/dx^2

    Дискретизация:
        p_i^{n+1} = p_i^n + dt * [
            0.5*beta * (p_i + x_i*(p_{i+1} - p_{i-1})/(2*dx))
            + 0.5*beta * (p_{i+1} - 2*p_i + p_{i-1}) / dx^2
        ]

    Начальное условие: p(x, 0) = delta-подобная (узкая гауссиана)
    Граничные условия: p = 0 на границах (Дирихле)

    Args:
        scheduler: Noise scheduler для получения beta(t).
        x_0: Начальная точка.
        x_range: Пространственный диапазон (x_min, x_max).
        nx: Количество пространственных точек.
        nt: Количество временных шагов.
        t_max: Максимальное время.

    Returns:
        (x_grid, t_grid, p_history):
            x_grid: пространственная сетка (nx,)
            t_grid: временная сетка (nt,)
            p_history: плотность на каждом шаге (nt, nx)
    """
    x_min, x_max = x_range
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / nt

    x_grid = torch.linspace(x_min, x_max, nx, dtype=torch.float64)
    t_grid = torch.linspace(0, t_max, nt, dtype=torch.float64)

    # Начальное условие: узкая гауссиана вокруг x_0
    # (аппроксимация дельта-функции)
    sigma_init = 0.05
    p = torch.exp(-0.5 * ((x_grid - x_0) / sigma_init) ** 2)
    p = p / (p.sum() * dx)  # нормализация

    p_history = torch.zeros(nt, nx, dtype=torch.float64)
    p_history[0] = p

    for n in range(1, nt):
        t = t_grid[n]
        beta_t = float(scheduler.beta(t.float()))

        # Конечные разности
        # dp/dx (центральные разности)
        dp_dx = torch.zeros_like(p)
        dp_dx[1:-1] = (p[2:] - p[:-2]) / (2 * dx)

        # d^2p/dx^2
        d2p_dx2 = torch.zeros_like(p)
        d2p_dx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dx ** 2)

        # Уравнение ФП для VP-SDE:
        # dp/dt = 0.5*beta * [p + x*dp/dx + d^2p/dx^2]
        dpdt = 0.5 * beta_t * (p + x_grid * dp_dx + d2p_dx2)

        p = p + dt * dpdt

        # Граничные условия
        p[0] = 0.0
        p[-1] = 0.0

        # Гарантируем неотрицательность
        p = p.clamp(min=0.0)

        p_history[n] = p

    return x_grid, t_grid, p_history


def analytical_gaussian_1d(
    scheduler: ScaledLinearScheduler,
    x_0: float,
    x_grid: Tensor,
    t: float,
) -> Tensor:
    """Аналитическое решение — гауссовское маргинальное распределение.

    p(x, t) = N(x; sqrt(alpha_bar(t))*x_0, 1-alpha_bar(t))
            = 1/sqrt(2*pi*sigma^2) * exp(-(x - mu)^2 / (2*sigma^2))

    Args:
        scheduler: Noise scheduler.
        x_0: Начальная точка.
        x_grid: Пространственная сетка.
        t: Время.

    Returns:
        Плотность вероятности на сетке x_grid.
    """
    t_tensor = torch.tensor(t, dtype=torch.float32)
    alpha_bar_t = float(scheduler.alpha_bar(t_tensor))

    mu = (alpha_bar_t ** 0.5) * x_0
    sigma2 = 1.0 - alpha_bar_t

    if sigma2 < 1e-10:
        # При t ≈ 0 возвращаем дельта-подобную функцию
        sigma2 = 1e-10

    x = x_grid.double()
    p = torch.exp(-0.5 * (x - mu) ** 2 / sigma2) / (2.0 * 3.141592653589793 * sigma2) ** 0.5
    return p


def verify_fokker_planck(
    scheduler: ScaledLinearScheduler,
    x_0: float = 0.0,
    check_times: list[float] | None = None,
    nx: int = 500,
    nt: int = 2000,
) -> dict[float, float]:
    """Верификация: сравнение численного решения ФП с аналитическим.

    Вычисляет L2 ошибку между численным решением уравнения ФП
    и аналитическим гауссовским распределением для нескольких
    моментов времени.

    Args:
        scheduler: Noise scheduler.
        x_0: Начальная точка.
        check_times: Моменты времени для проверки.
        nx: Количество пространственных точек.
        nt: Количество временных шагов.

    Returns:
        Словарь {t: L2_error} для каждого момента времени.
    """
    if check_times is None:
        check_times = [0.1, 0.25, 0.5, 0.75, 1.0]

    x_grid, t_grid, p_history = solve_fokker_planck_1d(
        scheduler, x_0=x_0, nx=nx, nt=nt
    )
    dx = float(x_grid[1] - x_grid[0])

    errors = {}
    for t_check in check_times:
        # Находим ближайший временной индекс
        idx = (t_grid - t_check).abs().argmin().item()
        p_numerical = p_history[idx]

        # Аналитическое решение
        p_analytical = analytical_gaussian_1d(scheduler, x_0, x_grid, t_check)

        # Нормализуем численное решение
        norm = p_numerical.sum() * dx
        if norm > 1e-10:
            p_numerical_normed = p_numerical / norm
        else:
            p_numerical_normed = p_numerical

        # L2 ошибка
        l2_error = float(torch.sqrt(((p_numerical_normed - p_analytical) ** 2).sum() * dx))
        errors[t_check] = l2_error

    return errors
