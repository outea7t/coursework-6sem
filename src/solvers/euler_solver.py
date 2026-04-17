# метод эйлера для probability flow ode диффузии


import torch
from torch import Tensor


class EulerSolver:

    def __init__(self, sde, num_steps: int = 30) -> None:
        self.sde = sde
        self.num_steps = num_steps

        # предвычисленные значения из дискретных alphas_cumprod
        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')

        if self._use_discrete:
            ac = sde.scheduler.alphas_cumprod.double()
            self._alpha_arr = torch.sqrt(ac).float()
            self._sigma_arr = torch.sqrt(1.0 - ac).float()

        self.timesteps = self._build_timesteps()

    def _build_timesteps(self) -> torch.Tensor:
        # равномерная сетка в пространстве дискретных timesteps
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            discrete = torch.linspace(N - 1, 0, self.num_steps + 1)
            discrete = discrete.round().long()
            continuous = discrete.float() / (N - 1)
            continuous = continuous.clamp(self.sde.t_min, self.sde.t_max)
            return continuous
        else:
            return torch.linspace(
                self.sde.t_max, self.sde.t_min, self.num_steps + 1
            )

    def reset(self) -> None:
        pass

    def _get_schedule(self, t: Tensor, device: torch.device):
        # alpha, sigma для времени t (из дискретного расписания)
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            idx = (t * (N - 1)).round().long().clamp(0, N - 1)
            idx_cpu = idx.cpu() if idx.is_cuda or str(idx.device).startswith('mps') else idx
            alpha = self._alpha_arr[idx_cpu].to(device)
            sigma = self._sigma_arr[idx_cpu].to(device)
        else:
            alpha, sigma = self.sde.marginal_params_at_t(t)
            alpha = alpha.to(device)
            sigma = sigma.to(device)
        return alpha, sigma

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        device = x.device

        alpha_t, sigma_t = self._get_schedule(t, device)
        alpha_prev, sigma_prev = self._get_schedule(t_prev, device)

        ndim = x.dim()
        alpha_t = alpha_t.reshape([1] * ndim)
        sigma_t = sigma_t.reshape([1] * ndim)
        alpha_prev = alpha_prev.reshape([1] * ndim)
        sigma_prev = sigma_prev.reshape([1] * ndim)

        # epsilon -> x_0
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        # последний шаг - возвращаем чистое x_0 (как в diffusers при sigma=0)
        is_last_step = float(t_prev) <= self.sde.t_min + 1e-6
        if is_last_step:
            return x0_pred

        # шаг эйлера в (sigma_hat, x_hat):
        #   x_hat_next = x_hat + (sigma_hat_prev - sigma_hat_t) * eps
        # после x_next = alpha_prev * x_hat_next эквивалент:
        #   x_next = alpha_prev * x0_pred + sigma_prev * eps
        x_next = alpha_prev * x0_pred + sigma_prev * model_output

        return x_next

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return 1

    @property
    def nfe_per_step(self) -> int:
        return 1
