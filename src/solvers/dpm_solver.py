# dpm-solver++ 2-го порядка для диффузионных ode

import torch
from torch import Tensor


class DPMSolverPP:

    def __init__(
        self,
        sde,
        num_steps: int = 20,
        solver_order: int = 2,
        thresholding: bool = False,
        dynamic_threshold_ratio: float = 0.995,
    ) -> None:
        self.sde = sde
        self.num_steps = num_steps
        self.solver_order = solver_order
        self.thresholding = thresholding
        self.dynamic_threshold_ratio = dynamic_threshold_ratio

        # предвычисленные значения из дискретных alphas_cumprod
        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')

        if self._use_discrete:
            ac = sde.scheduler.alphas_cumprod.double()
            self._alpha_arr = torch.sqrt(ac).float()
            self._sigma_arr = torch.sqrt(1.0 - ac).float()
            self._lambda_arr = torch.log(
                self._alpha_arr / self._sigma_arr.clamp(min=1e-20)
            )

        # история для мультишаговой экстраполяции
        self._x0_history: list[Tensor] = []
        self._lambda_history: list[Tensor] = []

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
        self._x0_history.clear()
        self._lambda_history.clear()

    def _get_schedule(self, t: Tensor, device: torch.device):
        # alpha, sigma, lambda для времени t
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            idx = (t * (N - 1)).round().long().clamp(0, N - 1)
            idx_cpu = idx.cpu() if idx.is_cuda or str(idx.device).startswith('mps') else idx
            alpha = self._alpha_arr[idx_cpu].to(device)
            sigma = self._sigma_arr[idx_cpu].to(device)
            lam = self._lambda_arr[idx_cpu].to(device)
        else:
            alpha, sigma = self.sde.marginal_params_at_t(t)
            alpha = alpha.to(device)
            sigma = sigma.to(device)
            lam = torch.log(alpha / sigma.clamp(min=1e-8)).to(device)
        return alpha, sigma, lam

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        device = x.device

        alpha_t, sigma_t, lambda_t = self._get_schedule(t, device)
        alpha_prev, sigma_prev, lambda_prev = self._get_schedule(t_prev, device)

        h = lambda_prev - lambda_t

        ndim = x.dim()
        alpha_t = alpha_t.reshape([1] * ndim)
        sigma_t = sigma_t.reshape([1] * ndim)
        alpha_prev = alpha_prev.reshape([1] * ndim)
        sigma_prev = sigma_prev.reshape([1] * ndim)

        # epsilon -> x0
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        if self.thresholding:
            x0_pred = self._dynamic_threshold(x0_pred)

        # последний шаг - возвращаем x0 напрямую
        is_last_step = float(t_prev) <= self.sde.t_min + 1e-6
        if is_last_step:
            return x0_pred

        self._x0_history.append(x0_pred.detach())
        self._lambda_history.append(lambda_t.squeeze().detach())

        # первый шаг - 1-й порядок, остальные - 2-й
        if len(self._x0_history) < 2 or self.solver_order == 1:
            x_out = self._dpm_solver_1(
                x, x0_pred, sigma_t, sigma_prev, alpha_prev, h
            )
        else:
            x_out = self._dpm_solver_2(
                x, x0_pred, sigma_t, sigma_prev, alpha_prev, h
            )

        while len(self._x0_history) > self.solver_order:
            self._x0_history.pop(0)
            self._lambda_history.pop(0)

        return x_out

    def _dpm_solver_1(
        self,
        x: Tensor,
        x0_pred: Tensor,
        sigma_t: Tensor,
        sigma_prev: Tensor,
        alpha_prev: Tensor,
        h: Tensor,
    ) -> Tensor:
        return (
            (sigma_prev / sigma_t) * x
            - alpha_prev * (torch.exp(-h) - 1.0) * x0_pred
        )

    def _dpm_solver_2(
        self,
        x: Tensor,
        x0_pred: Tensor,
        sigma_t: Tensor,
        sigma_prev: Tensor,
        alpha_prev: Tensor,
        h: Tensor,
    ) -> Tensor:
        D0 = x0_pred
        D0_prev = self._x0_history[-2]

        lam_s0 = self._lambda_history[-1]
        lam_s1 = self._lambda_history[-2]
        h_0 = lam_s0 - lam_s1

        r = h_0 / h
        D1 = (1.0 / r) * (D0 - D0_prev)

        exp_neg_h = torch.exp(-h)
        x_out = (
            (sigma_prev / sigma_t) * x
            - alpha_prev * (exp_neg_h - 1.0) * D0
            - 0.5 * alpha_prev * (exp_neg_h - 1.0) * D1
        )

        return x_out

    def _dynamic_threshold(self, x0_pred: Tensor) -> Tensor:
        batch_size = x0_pred.shape[0]
        x0_flat = x0_pred.reshape(batch_size, -1)

        s = torch.quantile(
            x0_flat.abs().float(), self.dynamic_threshold_ratio, dim=1
        )
        s = s.clamp(min=1.0)

        while s.dim() < x0_pred.dim():
            s = s.unsqueeze(-1)

        return x0_pred.clamp(-s, s) / s

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return self.solver_order

    @property
    def nfe_per_step(self) -> int:
        return 1
