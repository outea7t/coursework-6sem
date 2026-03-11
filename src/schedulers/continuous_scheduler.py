"""
Непрерывное расписание шума (Continuous Noise Schedule).

Работает напрямую с коэффициентами SDE в непрерывном времени.
Используется для формулировки через стохастические дифференциальные уравнения,
где beta(t) определяет мгновенную скорость добавления шума.

Математика:
    Прямое SDE (Variance Preserving):
        dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw

    Маргинальное распределение:
        q(x_t | x_0) = N(x_t; mean(t) * x_0, var(t) * I)

    где:
        mean(t) = exp(-0.5 * integral_0^t beta(s) ds) = sqrt(alpha_bar(t))
        var(t) = 1 - exp(-integral_0^t beta(s) ds) = 1 - alpha_bar(t)

    log-SNR:
        lambda(t) = log(mean(t)^2 / var(t))
                   = log(alpha_bar(t) / (1 - alpha_bar(t)))

    Этот scheduler оборачивает любой базовый scheduler и предоставляет
    непрерывные SDE коэффициенты.

Ссылки:
    Song, Y., et al. (2021).
    "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021).
"""

import torch

from .base_scheduler import BaseScheduler


class ContinuousScheduler(BaseScheduler):
    """Непрерывный scheduler для SDE формулировки.

    Оборачивает базовый scheduler, предоставляя непрерывные функции
    для drift и diffusion коэффициентов SDE.

    Args:
        base_scheduler: Базовый scheduler (linear, cosine, scaled_linear).
        num_train_timesteps: Количество дискретных шагов.
    """

    def __init__(
        self,
        base_scheduler: BaseScheduler | None = None,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        """
        Если base_scheduler задан, используются его beta(t) и alpha_bar(t).
        Иначе используется линейное расписание beta(t) = beta_min + t*(beta_max - beta_min)
        с параметрами, типичными для непрерывной SDE формулировки.

        Примечание: beta_min=0.1, beta_max=20.0 — стандартные значения для VP-SDE
        в статье Song et al. (2021), они отличаются от дискретных DDPM значений.
        """
        self.base_scheduler = base_scheduler
        self.beta_min_cont = beta_min
        self.beta_max_cont = beta_max
        super().__init__(num_train_timesteps)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Непрерывная функция beta(t).

        Если задан базовый scheduler — делегирует ему.
        Иначе: beta(t) = beta_min + t * (beta_max - beta_min)

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        if self.base_scheduler is not None:
            return self.base_scheduler.beta(t)
        return self.beta_min_cont + t * (self.beta_max_cont - self.beta_min_cont)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Кумулятивный коэффициент alpha_bar(t).

        Если задан базовый scheduler — делегирует ему.
        Иначе:
            integral_0^t beta(s) ds = beta_min*t + 0.5*(beta_max - beta_min)*t^2
            alpha_bar(t) = exp(-0.5 * integral)

        Примечание: множитель 0.5 перед интегралом появляется из-за формы
        VP-SDE: dx = -0.5*beta(t)*x*dt + ..., поэтому
        log(mean(t)) = -0.5 * integral_0^t beta(s) ds
        mean(t)^2 = alpha_bar(t) = exp(-integral_0^t beta(s) ds)

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение alpha_bar(t).
        """
        if self.base_scheduler is not None:
            return self.base_scheduler.alpha_bar(t)
        integral = self.beta_min_cont * t + 0.5 * (self.beta_max_cont - self.beta_min_cont) * t ** 2
        return torch.exp(-0.5 * integral)

    def drift_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """Коэффициент сноса (drift) для VP-SDE.

        f(t) = -0.5 * beta(t)

        В VP-SDE: dx = f(t)*x*dt + g(t)*dw
        где f(t) = -0.5*beta(t) — скалярный коэффициент при x.

        Args:
            t: Непрерывное время.

        Returns:
            Значение f(t) = -0.5 * beta(t).
        """
        return -0.5 * self.beta(t)

    def diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        """Коэффициент диффузии для VP-SDE.

        g(t) = sqrt(beta(t))

        Args:
            t: Непрерывное время.

        Returns:
            Значение g(t) = sqrt(beta(t)).
        """
        return torch.sqrt(self.beta(t))

    def mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Коэффициент среднего маргинального распределения.

        mean(t) = sqrt(alpha_bar(t))

        q(x_t | x_0) = N(x_t; mean(t)*x_0, sigma(t)^2*I)

        Args:
            t: Непрерывное время.

        Returns:
            sqrt(alpha_bar(t)).
        """
        return torch.sqrt(self.alpha_bar(t))

    def std_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Стандартное отклонение маргинального распределения.

        sigma(t) = sqrt(1 - alpha_bar(t))

        Args:
            t: Непрерывное время.

        Returns:
            sigma(t).
        """
        return torch.sqrt(1.0 - self.alpha_bar(t))
