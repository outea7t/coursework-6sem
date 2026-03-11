"""
Базовый абстрактный класс для noise schedulers.

Noise scheduler определяет функцию beta(t) — расписание добавления шума
в прямом диффузионном процессе. От выбора scheduler зависит:
- Скорость сходимости обратного процесса
- Качество генерируемых изображений
- Необходимое количество шагов сэмплирования

Математическое обоснование:
    В DDPM прямой процесс определяется как:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

    Маргинальное распределение:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    где alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)

    В непрерывном времени (t in [0, 1]):
        alpha_bar(t) = exp(-integral_0^t beta(s) ds)
"""

from abc import ABC, abstractmethod

import torch


class BaseScheduler(ABC):
    """Абстрактный базовый класс для noise schedulers.

    Каждый scheduler должен реализовать:
    - beta(t): мгновенное значение шума в момент t
    - alpha_bar(t): кумулятивное произведение (1 - beta_s) до момента t
    - discrete betas: дискретные значения beta для совместимости с U-Net
    """

    def __init__(self, num_train_timesteps: int = 1000) -> None:
        """
        Args:
            num_train_timesteps: Количество дискретных шагов при обучении модели.
        """
        self.num_train_timesteps = num_train_timesteps
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alphas_cumprod: torch.Tensor | None = None

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Непрерывная функция beta(t) — мгновенный уровень шума.

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        ...

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Кумулятивный коэффициент сохранения сигнала.

        Использует линейную интерполяцию предвычисленных дискретных
        alphas_cumprod для точного соответствия значениям, на которых
        обучена U-Net.

        Непрерывная формула alpha_bar(t) = exp(-integral beta(s) ds)
        систематически отличается от дискретного cumprod(1 - beta_i),
        что вызывает накопление ошибки в солверах и артефакты генерации.

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение alpha_bar(t).
        """
        t = torch.as_tensor(t, dtype=torch.float32)
        # Map t ∈ [0, 1] → index ∈ [0, N-1]
        idx = t * (self.num_train_timesteps - 1)
        idx = idx.clamp(0, self.num_train_timesteps - 1)

        idx_low = idx.long()
        idx_high = (idx_low + 1).clamp(max=self.num_train_timesteps - 1)
        frac = idx - idx_low.float()

        ac = self.alphas_cumprod
        return ac[idx_low] * (1.0 - frac) + ac[idx_high] * frac

    @property
    def betas(self) -> torch.Tensor:
        """Дискретные значения beta_t для t = 1, ..., T."""
        if self._betas is None:
            self._betas = self._compute_discrete_betas()
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        """Дискретные значения alpha_t = 1 - beta_t."""
        if self._alphas is None:
            self._alphas = 1.0 - self.betas
        return self._alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        """Кумулятивное произведение alpha_t: alpha_bar_t = prod_{s=1}^t alpha_s."""
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        return self._alphas_cumprod

    def _compute_discrete_betas(self) -> torch.Tensor:
        """Вычисляет дискретные beta значения из непрерывной функции beta(t).

        Использует равномерную сетку на [0, 1] с num_train_timesteps точками.

        Returns:
            Тензор дискретных beta значений формы (num_train_timesteps,).
        """
        t = torch.linspace(0, 1, self.num_train_timesteps, dtype=torch.float64)
        return self.beta(t).float()

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Стандартное отклонение шума в маргинальном распределении q(x_t | x_0).

        sigma(t) = sqrt(1 - alpha_bar(t))

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение sigma(t).
        """
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-Noise Ratio.

        SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение SNR(t).
        """
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Логарифм Signal-to-Noise Ratio.

        lambda(t) = log(SNR(t)) = log(alpha_bar(t)) - log(1 - alpha_bar(t))

        Это ключевая величина для DPM-Solver, так как ODE имеет простую
        форму в координатах lambda(t).

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение log_snr(t).
        """
        ab = self.alpha_bar(t)
        return torch.log(ab) - torch.log(1.0 - ab)
