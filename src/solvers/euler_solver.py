# метод эйлера для диффузии


import torch
from torch import Tensor


# классический метод эйлера x_{n+1} = x_n + f(t_n, x_n) * dt,
# применённый к обратной диффузии.
class EulerSolver:

    def __init__(self, sde, num_steps: int = 30) -> None:
        self.sde = sde
        self.num_steps = num_steps

        # если есть - берём alpha и sigma прямо из неё, чтобы точно
        # соответствовать тому, чему училась sdxl
        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')

        if self._use_discrete:
            # предвычисляем массивы alpha и sigma на все 1000 шагов.
            # считаем в double, сохраняем как float - чуть точнее, а
            # обращение по индексу работает очень быстро
            ac = sde.scheduler.alphas_cumprod.double()
            self._alpha_arr = torch.sqrt(ac).float()
            self._sigma_arr = torch.sqrt(1.0 - ac).float()

        # сетка моментов времени для всего обратного процесса
        self.timesteps = self._build_timesteps()

    # строит сетку моментов времени от t=1 (чистый шум) до t≈0 (картинка).
    # в дискретном режиме стараемся попадать ровно в точки сетки обучения
    # (0, 33, 66, ..., 999 для 30 шагов) - так шаги совпадают с тем,
    # что ожидает u-net
    def _build_timesteps(self) -> torch.Tensor:
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            # 31 точка от 999 до 0 включительно, округляем до целых
            discrete = torch.linspace(N - 1, 0, self.num_steps + 1)
            discrete = discrete.round().long()
            # переводим обратно в непрерывный t в [0, 1]
            continuous = discrete.float() / (N - 1)
            # страховочный clamp, чтобы не вылететь за [t_min, t_max] sde
            continuous = continuous.clamp(self.sde.t_min, self.sde.t_max)
            return continuous
        else:
            # непрерывный режим - равномерная сетка без привязки к дискретной
            return torch.linspace(
                self.sde.t_max, self.sde.t_min, self.num_steps + 1
            )

    # эйлер не держит внутреннего состояния между шагами,
    # но метод оставлен для совместимости с солверами, у которых оно есть
    def reset(self) -> None:
        pass

    # возвращает коэффициенты alpha и sigma в момент времени t.
    # в дискретном режиме - индексируемся в предвычисленные массивы,
    # в непрерывном - берём напрямую из sde
    def _get_schedule(self, t: Tensor, device: torch.device):
        if self._use_discrete:
            N = self.sde.scheduler.num_train_timesteps
            # непрерывный t в ближайший целый индекс 0..999
            idx = (t * (N - 1)).round().long().clamp(0, N - 1)
            # массивы лежат на cpu, индексация с mps/cuda-тензора
            # иногда ломается - принудительно переводим индекс на cpu
            idx_cpu = idx.cpu() if idx.is_cuda or str(idx.device).startswith('mps') else idx
            alpha = self._alpha_arr[idx_cpu].to(device)
            sigma = self._sigma_arr[idx_cpu].to(device)
        else:
            alpha, sigma = self.sde.marginal_params_at_t(t)
            alpha = alpha.to(device)
            sigma = sigma.to(device)
        return alpha, sigma

    # один шаг обратной диффузии: из x в момент t получаем x в момент t_prev.
    # принимает текущий зашумлённый латент, два момента времени и
    # предсказанный u-net шум epsilon (здесь это model_output)
    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        device = x.device

        # коэффициенты расписания в текущем и следующем моментах времени
        alpha_t, sigma_t = self._get_schedule(t, device)
        alpha_prev, sigma_prev = self._get_schedule(t_prev, device)

        # приводим скаляры к форме латента для корректного бродкаста
        # (латент 4-мерный, коэффициенты должны умножаться поэлементно)
        ndim = x.dim()
        alpha_t = alpha_t.reshape([1] * ndim)
        sigma_t = sigma_t.reshape([1] * ndim)
        alpha_prev = alpha_prev.reshape([1] * ndim)
        sigma_prev = sigma_prev.reshape([1] * ndim)

        # предсказываем чистую картинку x_0 из зашумлённого x и шума epsilon.
        # формула получается из x = alpha * x_0 + sigma * epsilon
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        # если это последний шаг (t_prev на границе) - отдаём предсказанный
        # x_0 напрямую. это совпадает с поведением diffusers при sigma=0
        # и избегает остаточного шума в финальной картинке
        is_last_step = float(t_prev) <= self.sde.t_min + 1e-6
        if is_last_step:
            return x0_pred

        # x_next = alpha_prev * x0_pred + sigma_prev * epsilon
        # переводим на следующий момент времени, где сигнала больше,
        # а шума меньше
        x_next = alpha_prev * x0_pred + sigma_prev * model_output

        return x_next

    # эйлер детерминированный - никакого случайного шума внутри шага.
    @property
    def is_stochastic(self) -> bool:
        return False

    # порядок точности метода: 1 - эйлер первого порядка.
    @property
    def order(self) -> int:
        return 1

    # сколько раз вызываем u-net за один шаг солвера.
    # у эйлера 1 вызов (только в цикле пайплайна),
    @property
    def nfe_per_step(self) -> int:
        return 1
