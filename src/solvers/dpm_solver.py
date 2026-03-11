"""
DPM-Solver++ — State-of-the-art солвер для диффузионных ODE.

Реализация соответствует diffusers' DPMSolverMultistepScheduler
(algorithm_type="dpmsolver++", solver_type="midpoint").

Ключевые отличия от наивного Euler ODE:
    - Работает в пространстве data prediction (x0), а не noise prediction
    - Использует экспоненциальный интегратор (exact solution для линейной части)
    - Мультишаговая экстраполяция для 2-го порядка при 1 NFE/шаг

Формулы (Lu et al., 2022):

    lambda(t) = log(alpha(t) / sigma(t))  — log signal-to-noise ratio

    DPM-Solver-1 (первого порядка, эквивалент DDIM):
        x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
                  - alpha_{t-1} * (exp(-h) - 1) * x0_pred

    DPM-Solver-2 (второго порядка, мультишаговый, midpoint):
        D0 = x0_pred(current)
        D1 = (1/r) * (D0 - D0_prev),  r = h_prev / h
        x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
                  - alpha_{t-1} * (exp(-h) - 1) * D0
                  - 0.5 * alpha_{t-1} * (exp(-h) - 1) * D1

    где h = lambda_{t-1} - lambda_t > 0 (lambda возрастает к чистому)

Ссылки:
    [1] Lu, C., et al. (2022). "DPM-Solver++: Fast Solver for Guided
        Sampling of Diffusion Probabilistic Models."
"""

import torch
from torch import Tensor


class DPMSolverPP:
    """DPM-Solver++ солвер.

    Args:
        sde: SDE модель (VPSDE).
        num_steps: Количество шагов.
        solver_order: Порядок солвера (1 или 2).
        thresholding: Применять dynamic thresholding для стабильности.
        dynamic_threshold_ratio: Квантиль для dynamic thresholding.
    """

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

        # Precompute schedule values from discrete alphas_cumprod.
        # This matches diffusers exactly and avoids continuous-time interpolation.
        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')

        if self._use_discrete:
            ac = sde.scheduler.alphas_cumprod.double()
            self._alpha_arr = torch.sqrt(ac).float()
            self._sigma_arr = torch.sqrt(1.0 - ac).float()
            self._lambda_arr = torch.log(
                self._alpha_arr / self._sigma_arr.clamp(min=1e-20)
            )

        # History for multistep extrapolation
        self._x0_history: list[Tensor] = []
        self._lambda_history: list[Tensor] = []

        # Build timestep grid
        self.timesteps = self._build_timesteps()

    def _build_timesteps(self) -> torch.Tensor:
        """Временная сетка, равномерная в пространстве дискретных timesteps.

        Соответствует diffusers' 'linspace' spacing:
            discrete = linspace(N-1, 0, num_steps+1).round()
            continuous = discrete / (N-1)

        Returns:
            Тензор (num_steps + 1,) от t_max к t_min.
        """
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            discrete = torch.linspace(N - 1, 0, self.num_steps + 1)
            discrete = discrete.round().long()
            continuous = discrete.float() / (N - 1)
            continuous = continuous.clamp(self.sde.t_min, self.sde.t_max)
            return continuous
        else:
            # Fallback for mock/test SDEs without discrete schedule
            return torch.linspace(
                self.sde.t_max, self.sde.t_min, self.num_steps + 1
            )

    def reset(self) -> None:
        """Сброс истории (вызывать перед новой генерацией)."""
        self._x0_history.clear()
        self._lambda_history.clear()

    def _get_schedule(self, t: Tensor, device: torch.device):
        """Получение alpha, sigma, lambda для времени t.

        При наличии дискретного расписания — точная выборка из
        precomputed массивов (без интерполяции). Иначе — вычисление
        через marginal_params_at_t.

        Returns:
            (alpha, sigma, lambda_val) — скалярные тензоры на device.
        """
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
        """Один шаг DPM-Solver++.

        Args:
            x: Текущие латенты.
            t: Текущее время (source, более шумное).
            t_prev: Следующее время (target, менее шумное).
            model_output: Предсказание шума epsilon.

        Returns:
            Латенты после одного шага.
        """
        device = x.device

        # Schedule values at source and target
        alpha_t, sigma_t, lambda_t = self._get_schedule(t, device)
        alpha_prev, sigma_prev, lambda_prev = self._get_schedule(t_prev, device)

        h = lambda_prev - lambda_t  # > 0 (lambda grows toward clean)

        # Reshape scalars for broadcasting with (batch, C, H, W)
        ndim = x.dim()
        alpha_t = alpha_t.reshape([1] * ndim)
        sigma_t = sigma_t.reshape([1] * ndim)
        alpha_prev = alpha_prev.reshape([1] * ndim)
        sigma_prev = sigma_prev.reshape([1] * ndim)

        # Convert epsilon prediction → x0 prediction
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        if self.thresholding:
            x0_pred = self._dynamic_threshold(x0_pred)

        # Final step: output x0 directly (like diffusers with final_sigmas_type="zero")
        is_last_step = float(t_prev) <= self.sde.t_min + 1e-6
        if is_last_step:
            return x0_pred

        # Save to history
        self._x0_history.append(x0_pred.detach())
        self._lambda_history.append(lambda_t.squeeze().detach())

        # Order selection: first step = 1st order, rest = 2nd order
        if len(self._x0_history) < 2 or self.solver_order == 1:
            x_out = self._dpm_solver_1(
                x, x0_pred, sigma_t, sigma_prev, alpha_prev, h
            )
        else:
            x_out = self._dpm_solver_2(
                x, x0_pred, sigma_t, sigma_prev, alpha_prev, h
            )

        # Keep only solver_order entries in history
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
        """DPM-Solver++ первого порядка (эквивалент DDIM).

        x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
                  - alpha_{t-1} * (exp(-h) - 1) * x0_pred
        """
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
        """DPM-Solver++ второго порядка (мультишаговый, midpoint).

        D0 = x0_pred (текущий)
        D1 = (1/r) * (D0 - D0_prev),  r = h_prev / h
        x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
                  - alpha_{t-1} * (exp(-h) - 1) * D0
                  - 0.5 * alpha_{t-1} * (exp(-h) - 1) * D1
        """
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
        """Dynamic thresholding для стабильности при высоком CFG."""
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
        return 1  # Мультишаговый — использует историю
