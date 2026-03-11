# variance preserving sde
# dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw

from typing import Tuple

import torch
from torch import Tensor

from ..schedulers.scaled_linear_scheduler import ScaledLinearScheduler


class VPSDE:

    def __init__(
        self,
        scheduler: ScaledLinearScheduler,
        t_min: float = 1e-3,
        t_max: float = 1.0,
    ) -> None:
        self.scheduler = scheduler
        self.t_min = t_min
        self.t_max = t_max

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        # f(x, t) = -0.5 * beta(t) * x
        beta_t = self.scheduler.beta(t)
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        # g(t) = sqrt(beta(t))
        return torch.sqrt(self.scheduler.beta(t))

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # q(x_t | x_0) = N(sqrt(alpha_bar) * x_0, (1 - alpha_bar) * I)
        alpha_bar_t = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar_t) * x_0
        std = torch.sqrt(1.0 - alpha_bar_t)
        return mean, std

    def marginal_params_at_t(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        # коэффициенты маргинала без x_0
        ab = self.scheduler.alpha_bar(t)
        ab = ab.clamp(min=1e-8)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1.0 - ab)
        return mean_coeff, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        return torch.randn(shape, device=device)

    def reverse_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        # f_reverse = f(x,t) - g(t)^2 * score
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - g ** 2 * score

    def reverse_ode_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        # probability flow ode: f - 0.5 * g^2 * score
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - 0.5 * g ** 2 * score

    def noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        # score = -epsilon / sigma(t)
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)

    def score_to_noise(self, score: Tensor, t: Tensor) -> Tensor:
        # epsilon = -score * sigma(t)
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < score.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -score * sigma_t
