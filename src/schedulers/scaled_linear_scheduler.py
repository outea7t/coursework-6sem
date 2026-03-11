"""
Масштабированное линейное расписание шума (Scaled Linear Noise Schedule).

Используется по умолчанию в Stable Diffusion. Отличие от обычного линейного:
линейная интерполяция производится в пространстве sqrt(beta), что обеспечивает
более плавное нарастание шума.

Математика:
    sqrt(beta(t)) = sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min))
    beta(t) = (sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min)))^2

    Обозначим: a = sqrt(beta_min), b = sqrt(beta_max)
    beta(t) = (a + t*(b - a))^2 = a^2 + 2*a*(b-a)*t + (b-a)^2 * t^2

    Интеграл beta(t) по [0, t]:
        integral_0^t beta(s) ds = a^2 * t + a*(b-a)*t^2 + (b-a)^2 * t^3 / 3

    alpha_bar(t) = exp(-integral_0^t beta(s) ds)

Параметры по умолчанию: beta_min = 0.00085, beta_max = 0.012
(значения из Stable Diffusion).

Дискретные значения:
    В DDPM прямой процесс определяется как:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

    Маргинальное распределение:
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    где alpha_bar_t = prod_{s=1}^{t} (1 - beta_s)

Ссылки:
    Rombach, R., et al. (2022).
    "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022).
"""

import math

import torch


class ScaledLinearScheduler:
    """Масштабированное линейное расписание шума.

    Линейная интерполяция в пространстве sqrt(beta):
        sqrt(beta(t)) = sqrt(beta_min) + t * (sqrt(beta_max) - sqrt(beta_min))

    Args:
        beta_min: Минимальное значение beta.
        beta_max: Максимальное значение beta.
        num_train_timesteps: Количество дискретных шагов.
    """

    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
    ) -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sqrt_beta_min = math.sqrt(beta_min)
        self.sqrt_beta_max = math.sqrt(beta_max)
        self.num_train_timesteps = num_train_timesteps

        # Lazy-computed discrete values
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alphas_cumprod: torch.Tensor | None = None

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Масштабированная линейная функция beta(t) (непрерывная).

        Дискретная формула: beta_discrete(t) = (sqrt(beta_min) + t*(sqrt(beta_max) - sqrt(beta_min)))^2
        Непрерывная формула: beta(t) = N * beta_discrete(t)

        Масштабирование на N = num_train_timesteps необходимо, чтобы
        integral_0^1 beta(s) ds совпадал с суммой дискретных бет.

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение beta(t).
        """
        sqrt_beta = self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min)
        return self.num_train_timesteps * sqrt_beta ** 2

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Кумулятивный коэффициент сохранения сигнала.

        Использует линейную интерполяцию предвычисленных дискретных
        alphas_cumprod для точного соответствия значениям, на которых
        обучена U-Net.

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение alpha_bar(t).
        """
        t = torch.as_tensor(t, dtype=torch.float32)
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
        """Дискретные beta: линейная сетка в пространстве sqrt(beta)."""
        sqrt_betas = torch.linspace(
            self.sqrt_beta_min,
            self.sqrt_beta_max,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        return sqrt_betas ** 2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Стандартное отклонение шума: sigma(t) = sqrt(1 - alpha_bar(t)).

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение sigma(t).
        """
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-Noise Ratio: SNR(t) = alpha_bar(t) / (1 - alpha_bar(t)).

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

        Args:
            t: Непрерывное время, t in [0, 1].

        Returns:
            Значение log_snr(t).
        """
        ab = self.alpha_bar(t)
        return torch.log(ab) - torch.log(1.0 - ab)
