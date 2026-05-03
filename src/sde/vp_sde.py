# здесь живёт само уравнение, по которому шумится картинка
# (вариант с сохранением дисперсии, VP-SDE).
# функции в классе - это то, что нам нужно для прямого процесса
# (картинка -> шум) и обратного (шум -> картинка).
# конкретные формулы все расписаны в теоретической части курсовой

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
        # расписание задаёт коэффициенты beta(t), alpha_bar(t) и т.п.
        self.scheduler = scheduler
        # границы по времени. ровно в t=0 у нас деление на ноль
        # в нескольких формулах, поэтому реально работаем с [t_min, t_max]
        self.t_min = t_min
        self.t_max = t_max

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        # часть прямого уравнения, отвечающая за то,
        # что картинка постепенно "выцветает"
        beta_t = self.scheduler.beta(t)
        # подгоняем число осей beta_t под x,
        # чтобы умножение шло поэлементно
        while beta_t.dim() < x.dim():
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        # вторая часть уравнения - то, что стоит перед случайным членом
        # (т.е. контролирует, насколько сильно изменяется процесс)
        return torch.sqrt(self.scheduler.beta(t))

    def marginal_params(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # отвечает на вопрос "если зашумлять картинку x_0 в течение
        # времени t, во что она превратится в среднем".
        # ответ - гауссово распределение со средним и отклонением,
        # которые мы и возвращаем

        # clamp нужен на случай, если t близок к 1 - тогда alpha_bar
        # очень близок к нулю, и без защиты получится деление на ноль
        alpha_bar_t = self.scheduler.alpha_bar(t).clamp(min=1e-8)
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        mean = torch.sqrt(alpha_bar_t) * x_0
        std = torch.sqrt(1.0 - alpha_bar_t)
        return mean, std

    def marginal_params_at_t(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        # то же самое, но без исходной картинки - только коэффициенты.
        # пригодится в шаге Эйлера, где x_0 нам и неизвестен
        ab = self.scheduler.alpha_bar(t)
        ab = ab.clamp(min=1e-8)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1.0 - ab)
        return mean_coeff, std

    def prior_sampling(self, shape: Tuple[int, ...], device: str = "cpu") -> Tensor:
        # отправная точка обратного процесса - просто гауссов шум
        return torch.randn(shape, device=device)

    def reverse_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        # правая часть обратного уравнения, если решать его в случайном виде.
        # в самой генерации мы это не используем - оставлено для полноты
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - g ** 2 * score

    def reverse_ode_drift(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        # правая часть детерминированной (без случайности) версии
        # обратного уравнения. отличие от случайной версии:
        # перед score стоит 0.5, и нет случайного члена.
        # именно это уравнение и решает метод Эйлера - так
        # одинаковый seed всегда даёт одинаковую картинку
        f = self.drift(x, t)
        g = self.diffusion(t)
        while g.dim() < x.dim():
            g = g.unsqueeze(-1)
        return f - 0.5 * g ** 2 * score

    def noise_to_score(self, noise: Tensor, t: Tensor) -> Tensor:
        # u-net выдаёт шум, а в обратном уравнении нужен не сам шум,
        # а так называемая функция оценки. между ними простой пересчёт
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < noise.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -noise / sigma_t.clamp(min=1e-8)

    def score_to_noise(self, score: Tensor, t: Tensor) -> Tensor:
        # пересчёт в обратную сторону - из функции оценки в шум.
        # в самой генерации не нужен, использую в проверочных скриптах
        _, sigma_t = self.marginal_params_at_t(t)
        while sigma_t.dim() < score.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        return -score * sigma_t
