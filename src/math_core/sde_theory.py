# теоретические функции для анализа свойств sde

import torch
from torch import Tensor

from ..schedulers.scaled_linear_scheduler import ScaledLinearScheduler


def transition_kernel_params(
    scheduler: ScaledLinearScheduler, t: Tensor
) -> tuple[Tensor, Tensor]:
    # параметры переходного ядра q(x_t | x_0)
    alpha_bar_t = scheduler.alpha_bar(t)
    mean_coeff = torch.sqrt(alpha_bar_t)
    variance = 1.0 - alpha_bar_t
    return mean_coeff, variance


def signal_to_noise_ratio(scheduler: ScaledLinearScheduler, t: Tensor) -> Tensor:
    ab = scheduler.alpha_bar(t)
    return ab / (1.0 - ab).clamp(min=1e-8)


def log_signal_to_noise_ratio(scheduler: ScaledLinearScheduler, t: Tensor) -> Tensor:
    ab = scheduler.alpha_bar(t).clamp(min=1e-8, max=1.0 - 1e-8)
    return torch.log(ab) - torch.log(1.0 - ab)


def discrete_to_continuous_beta(
    discrete_betas: Tensor, num_train_timesteps: int = 1000
) -> tuple[float, float]:
    T = num_train_timesteps
    beta_min_cont = float(discrete_betas[0]) * T
    beta_max_cont = float(discrete_betas[-1]) * T
    return beta_min_cont, beta_max_cont


def noise_level_at_timestep(
    scheduler: ScaledLinearScheduler, t: Tensor
) -> dict[str, Tensor]:
    ab = scheduler.alpha_bar(t)
    return {
        "alpha_bar": ab,
        "sigma": torch.sqrt(1.0 - ab),
        "snr": ab / (1.0 - ab).clamp(min=1e-8),
        "log_snr": torch.log(ab.clamp(min=1e-8)) - torch.log((1.0 - ab).clamp(min=1e-8)),
        "beta": scheduler.beta(t),
    }


def optimal_timestep_spacing(
    scheduler: ScaledLinearScheduler,
    num_steps: int,
    spacing: str = "uniform",
) -> Tensor:
    if spacing == "uniform":
        return torch.linspace(1.0, 1e-3, num_steps + 1)

    elif spacing == "log_snr":
        # равномерная сетка в пространстве log-snr
        t_grid = torch.linspace(0.001, 0.999, 1000)
        log_snr_grid = log_signal_to_noise_ratio(scheduler, t_grid)

        lambda_max = log_snr_grid[0]
        lambda_min = log_snr_grid[-1]
        target_lambdas = torch.linspace(lambda_min, lambda_max, num_steps + 1)

        timesteps = []
        for lam in target_lambdas:
            idx = (log_snr_grid - lam).abs().argmin()
            timesteps.append(t_grid[idx])
        return torch.stack(timesteps)

    elif spacing == "quadratic":
        s = torch.linspace(1.0, 0.0, num_steps + 1)
        return s ** 2 * (1.0 - 1e-3) + 1e-3

    else:
        raise ValueError(f"Unknown spacing: {spacing}")
