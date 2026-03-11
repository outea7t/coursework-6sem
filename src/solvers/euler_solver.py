# метод эйлера для probability flow ode
# x_{t-1} = x_t + f_ode(x, t) * dt
# где f_ode = f(x,t) - 0.5 * g(t)^2 * score(x,t)

import torch
from torch import Tensor


class EulerODESolver:

    def __init__(self, sde, num_steps: int = 30) -> None:
        self.sde = sde
        self.num_steps = num_steps

        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')
        self.timesteps = self._build_timesteps()

    def _build_timesteps(self) -> torch.Tensor:
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

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        dt = t_prev - t

        # epsilon -> score
        score = self.sde.noise_to_score(model_output, t)

        # probability flow ode drift
        drift = self.sde.reverse_ode_drift(x, t, score)

        # шаг эйлера
        x_next = x + drift * dt

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
