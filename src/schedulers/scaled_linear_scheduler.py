# масштабированное линейное расписание шума
#
# расписание - это функция beta(t), которая задаёт, сколько шума
# добавлять на каждом моменте времени t из отрезка [0, 1].
# здесь используется тот же вариант, на котором обучали sdxl:
# линейная интерполяция не самой beta, а её квадратного корня.
#
# из beta(t) получаем два главных коэффициента:
#   alpha_bar(t) - доля сохранённого "сигнала" в момент t (от 1 до ~0)
#   sigma(t)     - уровень шума в момент t (от ~0 до 1)
#
# при t=0 у нас чистая картинка (alpha_bar=1, sigma=0),
# при t=1 - чистый шум (alpha_bar=0, sigma=1)

import math

import torch


class ScaledLinearScheduler:

    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
    ) -> None:
        # значения beta_min и beta_max взяты из обучения sdxl - менять их нельзя,
        # иначе расписание перестанет совпадать с тем, чему училась нейросеть
        self.beta_min = beta_min
        self.beta_max = beta_max
        # квадратные корни считаем один раз и сохраняем -
        # они нужны в beta(t) при каждом вызове
        self.sqrt_beta_min = math.sqrt(beta_min)
        self.sqrt_beta_max = math.sqrt(beta_max)
        # размер дискретной сетки времени - 1000 шагов (с этим обучалась sdxl)
        self.num_train_timesteps = num_train_timesteps

        # ленивые кэши - считаются только при первом обращении,
        # потом отдаются готовыми. самый дорогой - alphas_cumprod
        # (последовательное перемножение 1000 чисел)
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alphas_cumprod: torch.Tensor | None = None

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # вычисление beta(t) для непрерывного t из [0, 1].
        # умножение на num_train_timesteps - чтобы перейти от
        # дискретного шага к непрерывному времени
        sqrt_beta = self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min)
        return self.num_train_timesteps * sqrt_beta ** 2

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        # alpha_bar(t) - доля сохранённого сигнала в момент времени t.
        # это главная величина расписания.
        #
        # значения alpha_bar для целых шагов (0, 1, ..., 999) считаются
        # один раз и хранятся в массиве alphas_cumprod.
        # для произвольного t из [0, 1] делаем линейную интерполяцию
        # между двумя соседними точками таблицы.

        t = torch.as_tensor(t, dtype=torch.float32)
        # переводим t из [0, 1] в "индекс" по таблице длиной 1000
        idx = t * (self.num_train_timesteps - 1)
        idx = idx.clamp(0, self.num_train_timesteps - 1)

        # два соседних целых индекса и дробная часть между ними.
        # пример: t=0.5 -> idx=499.5, idx_low=499, idx_high=500, frac=0.5
        idx_low = idx.long()
        idx_high = (idx_low + 1).clamp(max=self.num_train_timesteps - 1)
        frac = idx - idx_low.float()

        # обычная линейная интерполяция между двумя точками
        ac = self.alphas_cumprod
        return ac[idx_low] * (1.0 - frac) + ac[idx_high] * frac

    @property
    def betas(self) -> torch.Tensor:
        # таблица из 1000 значений beta - по одному на каждый шаг.
        # считается один раз при первом обращении, потом отдаётся из кэша
        if self._betas is None:
            self._betas = self._compute_discrete_betas()
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        # доля сигнала, которая остаётся за один маленький шаг
        if self._alphas is None:
            self._alphas = 1.0 - self.betas
        return self._alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        # накопленное произведение alpha - сколько сигнала осталось
        # от исходной картинки к моменту t. именно эту таблицу
        # использует alpha_bar(t) как опорные точки для интерполяции
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        return self._alphas_cumprod

    def _compute_discrete_betas(self) -> torch.Tensor:
        # 1000 равномерно распределённых значений sqrt(beta)
        # от sqrt(beta_min) до sqrt(beta_max). дальше возводим в квадрат -
        # получаем сами beta. именно так делает sdxl
        sqrt_betas = torch.linspace(
            self.sqrt_beta_min,
            self.sqrt_beta_max,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        return sqrt_betas ** 2

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        # уровень шума в момент t
        return torch.sqrt(1.0 - self.alpha_bar(t))

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        # отношение сигнал/шум.
        # в момент t=0 (чистая картинка) стремится к бесконечности,
        # в момент t=1 (чистый шум) стремится к нулю
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab)

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        # логарифм от snr - удобная "логарифмическая" шкала времени.
        # в основном цикле не используется, нужен для математической верификации
        ab = self.alpha_bar(t)
        return torch.log(ab) - torch.log(1.0 - ab)
