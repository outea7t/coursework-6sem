"""
Теоретические функции для анализа свойств SDE.

Этот модуль содержит функции для вычисления аналитических свойств
диффузионных процессов — переходные ядра, SNR, связь между
непрерывной и дискретной формулировками.

Используется для:
1. Верификации корректности реализации SDE и schedulers
2. Визуализации в Jupyter notebooks
3. Теоретической части курсовой работы

Ключевые концепции:
===================

1. Переходное ядро (Transition Kernel):
   q(x_t | x_0) — условное распределение зашумлённого состояния
   при заданном начальном.

2. Signal-to-Noise Ratio (SNR):
   SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
   Ключевая величина, определяющая качество генерации.

3. log-SNR (lambda):
   lambda(t) = log(SNR(t))
   Используется в DPM-Solver для оптимальной параметризации ODE.

4. Связь дискретного и непрерывного:
   Дискретный DDPM с T шагами → непрерывное SDE при T → inf.
"""

import torch
from torch import Tensor

from ..schedulers.scaled_linear_scheduler import ScaledLinearScheduler


def transition_kernel_params(
    scheduler: ScaledLinearScheduler, t: Tensor
) -> tuple[Tensor, Tensor]:
    """Параметры переходного ядра q(x_t | x_0) для VP-SDE.

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar(t)) * x_0, (1 - alpha_bar(t)) * I)

    Args:
        scheduler: Noise scheduler.
        t: Непрерывное время.

    Returns:
        (mean_coeff, variance):
            mean_coeff = sqrt(alpha_bar(t)), множитель при x_0
            variance = 1 - alpha_bar(t), дисперсия шума
    """
    alpha_bar_t = scheduler.alpha_bar(t)
    mean_coeff = torch.sqrt(alpha_bar_t)
    variance = 1.0 - alpha_bar_t
    return mean_coeff, variance


def signal_to_noise_ratio(scheduler: ScaledLinearScheduler, t: Tensor) -> Tensor:
    """Signal-to-Noise Ratio (SNR).

    SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))

    Свойства:
    - SNR(0) = inf (чистый сигнал)
    - SNR(T) -> 0 (чистый шум)
    - SNR монотонно убывает

    Физический смысл: отношение энергии сигнала к энергии шума.

    Args:
        scheduler: Noise scheduler.
        t: Время.

    Returns:
        SNR(t).
    """
    ab = scheduler.alpha_bar(t)
    return ab / (1.0 - ab).clamp(min=1e-8)


def log_signal_to_noise_ratio(scheduler: ScaledLinearScheduler, t: Tensor) -> Tensor:
    """Логарифм Signal-to-Noise Ratio.

    lambda(t) = log(SNR(t)) = log(alpha_bar(t)) - log(1 - alpha_bar(t))

    Это ключевая переменная для DPM-Solver:
    - ODE имеет простую форму в координатах lambda
    - Равномерная сетка по lambda дает оптимальные шаги
    - Используется для мультишаговой экстраполяции

    Args:
        scheduler: Noise scheduler.
        t: Время.

    Returns:
        lambda(t) = log_SNR(t).
    """
    ab = scheduler.alpha_bar(t).clamp(min=1e-8, max=1.0 - 1e-8)
    return torch.log(ab) - torch.log(1.0 - ab)


def discrete_to_continuous_beta(
    discrete_betas: Tensor, num_train_timesteps: int = 1000
) -> tuple[float, float]:
    """Конвертация дискретных beta в параметры непрерывного линейного расписания.

    Связь между дискретным и непрерывным:
        beta_discrete_t ≈ beta_continuous(t/T) / T

    Для линейного расписания:
        beta_continuous(t) = beta_min + t*(beta_max - beta_min)
        beta_discrete_t = beta_continuous(t/T) / T

    Args:
        discrete_betas: Дискретные beta, форма (T,).
        num_train_timesteps: T.

    Returns:
        (beta_min_continuous, beta_max_continuous).
    """
    T = num_train_timesteps
    beta_min_cont = float(discrete_betas[0]) * T
    beta_max_cont = float(discrete_betas[-1]) * T
    return beta_min_cont, beta_max_cont


def noise_level_at_timestep(
    scheduler: ScaledLinearScheduler, t: Tensor
) -> dict[str, Tensor]:
    """Полная информация об уровне шума в момент t.

    Возвращает все ключевые параметры для анализа и визуализации.

    Args:
        scheduler: Noise scheduler.
        t: Время.

    Returns:
        Словарь с ключами:
            alpha_bar: кумулятивный alpha
            sigma: стандартное отклонение шума
            snr: signal-to-noise ratio
            log_snr: логарифм SNR
            beta: мгновенное значение beta
    """
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
    """Вычисление оптимального расположения временных шагов.

    Разные стратегии:
    - "uniform": равномерная сетка по t
    - "log_snr": равномерная сетка по log-SNR (оптимально для DPM-Solver)
    - "quadratic": квадратичная сетка (больше шагов в начале)

    Args:
        scheduler: Noise scheduler.
        num_steps: Количество шагов.
        spacing: Стратегия размещения.

    Returns:
        Тензор временных шагов от T до epsilon.
    """
    if spacing == "uniform":
        return torch.linspace(1.0, 1e-3, num_steps + 1)

    elif spacing == "log_snr":
        # Равномерная сетка в пространстве log-SNR
        t_grid = torch.linspace(0.001, 0.999, 1000)
        log_snr_grid = log_signal_to_noise_ratio(scheduler, t_grid)

        # lambda монотонно убывает: от log_snr(0.001) до log_snr(0.999)
        lambda_max = log_snr_grid[0]
        lambda_min = log_snr_grid[-1]
        target_lambdas = torch.linspace(lambda_min, lambda_max, num_steps + 1)

        # Найти t для каждого целевого lambda
        timesteps = []
        for lam in target_lambdas:
            idx = (log_snr_grid - lam).abs().argmin()
            timesteps.append(t_grid[idx])
        return torch.stack(timesteps)

    elif spacing == "quadratic":
        # Больше шагов в начале (ближе к t=T), где изменения сильнее
        s = torch.linspace(1.0, 0.0, num_steps + 1)
        return s ** 2 * (1.0 - 1e-3) + 1e-3

    else:
        raise ValueError(f"Unknown spacing: {spacing}")
