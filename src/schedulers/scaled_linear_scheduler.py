# масштабированное линейное расписание шума
# линейная интерполяция в пространстве sqrt(beta)

import math

import torch


class ScaledLinearScheduler:

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

        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alphas_cumprod: torch.Tensor | None = None

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # непрерывная beta, масштабированная на N
        sqrt_beta = self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min)
        return self.num_train_timesteps * sqrt_beta ** 2

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        # интерполяция предвычисленных дискретных alphas_cumprod
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
        if self._betas is None:
            self._betas = self._compute_discrete_betas()
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        if self._alphas is None:
            self._alphas = 1.0 - self.betas
        return self._alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        return self._alphas_cumprod

    def _compute_discrete_betas(self) -> torch.Tensor:
        sqrt_betas = torch.linspace(
            self.sqrt_beta_min,
            self.sqrt_beta_max,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        return sqrt_betas ** 2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        # signal-to-noise ratio
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar(t)
        return torch.log(ab) - torch.log(1.0 - ab)
