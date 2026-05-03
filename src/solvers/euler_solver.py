# метод Эйлера для обыкновенного дифференциального уравнения обратного процесса
#
# конкретные формулы (само уравнение, шаг метода Эйлера и связь с alpha/sigma)
# приведены в теоретической части курсовой работы.
#
# идея реализации: на каждом шаге сначала из текущего зашумлённого x
# и предсказанного шума eps восстанавливаем "что должно было быть" -
# чистую картинку x_0. потом снова чуть-чуть зашумляем её, но уже
# до меньшего уровня шума. математически это эквивалентно прямой
# записи метода Эйлера, но численно устойчивее.

import torch
from torch import Tensor


class EulerSolver:

    def __init__(self, sde, num_steps: int = 30) -> None:
        # ссылка на стохастическое уравнение - оттуда возьмём расписание
        self.sde = sde
        # сколько шагов метода Эйлера выполнить за всю генерацию
        self.num_steps = num_steps

        # если у расписания есть готовая таблица alphas_cumprod -
        # берём alpha и sigma прямо из неё, чтобы значения точно
        # совпадали с тем, на чём обучали u-net
        self._use_discrete = hasattr(sde.scheduler, 'alphas_cumprod')

        if self._use_discrete:
            # заранее считаем alpha и sigma для всех 1000 шагов.
            # делаем это в double для точности, сохраняем как float.
            # потом обращение по индексу очень быстрое
            ac = sde.scheduler.alphas_cumprod.double()
            self._alpha_arr = torch.sqrt(ac).float()
            self._sigma_arr = torch.sqrt(1.0 - ac).float()

        # сетка моментов времени для всего обратного процесса -
        # массив из (num_steps + 1) точек, считаем заранее
        self.timesteps = self._build_timesteps()

    def _build_timesteps(self) -> torch.Tensor:
        # сетка моментов времени, по которой мы будем идти от шума к картинке.
        # начинаем с t=1 (чистый шум), заканчиваем на t≈0 (готовая картинка).
        # число точек = num_steps + 1, потому что N шагов делят отрезок на N+1 узлов

        if self._use_discrete:
            # дискретный режим: подбираем точки так, чтобы они
            # точно попадали в "родную" сетку обучения u-net (0, 1, ..., 999)
            N = self.sde.scheduler.num_train_timesteps
            # 31 точка от 999 до 0 включительно
            discrete = torch.linspace(N - 1, 0, self.num_steps + 1)
            # округляем до целых - чтобы не "застрять" между табличными значениями
            discrete = discrete.round().long()
            # обратно в непрерывный t из [0, 1]
            continuous = discrete.float() / (N - 1)
            # подстраховываемся, чтобы не выйти за рабочий диапазон уравнения
            continuous = continuous.clamp(self.sde.t_min, self.sde.t_max)
            return continuous
        else:
            # непрерывный режим: равномерная сетка от t_max до t_min
            return torch.linspace(
                self.sde.t_max, self.sde.t_min, self.num_steps + 1
            )

    def reset(self) -> None:
        # метод Эйлера не запоминает ничего между шагами,
        # но метод оставлен для единообразия интерфейса
        pass

    def _get_schedule(self, t: Tensor, device: torch.device):
        # получение коэффициентов alpha и sigma в момент времени t.
        # это обёртка над двумя режимами работы (дискретным и непрерывным)

        if self._use_discrete:
            # переводим непрерывный t из [0, 1] в ближайший
            # целый индекс из диапазона 0..999
            N = self.sde.scheduler.num_train_timesteps
            idx = (t * (N - 1)).round().long().clamp(0, N - 1)

            # таблицы alpha и sigma лежат на cpu. на видеокарте apple (mps)
            # индексация cpu-массива тензором с устройства иногда даёт сбой -
            # поэтому насильно переводим индекс на cpu
            if idx.is_cuda or str(idx.device).startswith('mps'):
                idx_cpu = idx.cpu()
            else:
                idx_cpu = idx

            alpha = self._alpha_arr[idx_cpu].to(device)
            sigma = self._sigma_arr[idx_cpu].to(device)
        else:
            # в непрерывном режиме просто спрашиваем у уравнения
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
        # один шаг обратной диффузии методом Эйлера.
        # на входе:
        #   x - текущее зашумлённое скрытое представление в момент t;
        #   t - текущий момент времени;
        #   t_prev - следующий момент времени (ближе к нулю, t_prev < t);
        #   model_output - предсказанный u-net'ом шум eps в текущем x.
        # на выходе - новое скрытое представление в момент t_prev,
        # уровень шума в нём меньше

        device = x.device

        # коэффициенты расписания в текущем и следующем моментах времени
        alpha_t, sigma_t = self._get_schedule(t, device)
        alpha_prev, sigma_prev = self._get_schedule(t_prev, device)

        # приводим скаляры к форме картинки (4 оси: батч, каналы, высота, ширина) -
        # чтобы умножение работало поэлементно
        ndim = x.dim()
        alpha_t = alpha_t.reshape([1] * ndim)
        sigma_t = sigma_t.reshape([1] * ndim)
        alpha_prev = alpha_prev.reshape([1] * ndim)
        sigma_prev = sigma_prev.reshape([1] * ndim)

        # выражаем чистую картинку x_0, зная зашумлённое x и шум eps.
        # clamp(1e-8) - защита от деления на ноль при alpha близком к 0
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        # если это последний шаг (t_prev на самой границе) - возвращаем
        # сразу x_0. иначе на финале остаётся остаточный шум и картинка
        # получается чуть зернистая
        is_last_step = float(t_prev) <= self.sde.t_min + 1e-6
        if is_last_step:
            return x0_pred

        # переводим предсказанное x_0 на следующий момент времени:
        # шума там должно быть меньше, поэтому используем коэффициенты
        # alpha_prev и sigma_prev
        x_next = alpha_prev * x0_pred + sigma_prev * model_output

        return x_next

    @property
    def is_stochastic(self) -> bool:
        # метод детерминированный - случайного шума внутри шага нет
        return False

    @property
    def order(self) -> int:
        # порядок точности метода: метод Эйлера - первого порядка
        return 1

    @property
    def nfe_per_step(self) -> int:
        # количество вызовов нейронной сети за один шаг.
        # у метода Эйлера - один вызов на шаг
        return 1
