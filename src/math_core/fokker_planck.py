# численная верификация уравнения фоккера-планка для vp-sde

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
    x_min, x_max = x_range
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / nt

    x_grid = torch.linspace(x_min, x_max, nx, dtype=torch.float64)
    t_grid = torch.linspace(0, t_max, nt, dtype=torch.float64)

    # начальное условие - узкая гауссиана
    sigma_init = 0.05
    p = torch.exp(-0.5 * ((x_grid - x_0) / sigma_init) ** 2)
    p = p / (p.sum() * dx)

    p_history = torch.zeros(nt, nx, dtype=torch.float64)
    p_history[0] = p

    for n in range(1, nt):
        t = t_grid[n]
        beta_t = float(scheduler.beta(t.float()))

        dp_dx = torch.zeros_like(p)
        dp_dx[1:-1] = (p[2:] - p[:-2]) / (2 * dx)

        d2p_dx2 = torch.zeros_like(p)
        d2p_dx2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dx ** 2)

        # dp/dt = 0.5*beta * [p + x*dp/dx + d2p/dx2]
        dpdt = 0.5 * beta_t * (p + x_grid * dp_dx + d2p_dx2)

        p = p + dt * dpdt
        p[0] = 0.0
        p[-1] = 0.0
        p = p.clamp(min=0.0)

        p_history[n] = p

    return x_grid, t_grid, p_history


def analytical_gaussian_1d(
    scheduler: ScaledLinearScheduler,
    x_0: float,
    x_grid: Tensor,
    t: float,
) -> Tensor:
    # аналитическое гауссовское маргинальное распределение
    t_tensor = torch.tensor(t, dtype=torch.float32)
    alpha_bar_t = float(scheduler.alpha_bar(t_tensor))

    mu = (alpha_bar_t ** 0.5) * x_0
    sigma2 = 1.0 - alpha_bar_t

    if sigma2 < 1e-10:
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
    # сравнение численного решения с аналитическим
    if check_times is None:
        check_times = [0.1, 0.25, 0.5, 0.75, 1.0]

    x_grid, t_grid, p_history = solve_fokker_planck_1d(
        scheduler, x_0=x_0, nx=nx, nt=nt
    )
    dx = float(x_grid[1] - x_grid[0])

    errors = {}
    for t_check in check_times:
        idx = (t_grid - t_check).abs().argmin().item()
        p_numerical = p_history[idx]

        p_analytical = analytical_gaussian_1d(scheduler, x_0, x_grid, t_check)

        norm = p_numerical.sum() * dx
        if norm > 1e-10:
            p_numerical_normed = p_numerical / norm
        else:
            p_numerical_normed = p_numerical

        l2_error = float(torch.sqrt(((p_numerical_normed - p_analytical) ** 2).sum() * dx))
        errors[t_check] = l2_error

    return errors
