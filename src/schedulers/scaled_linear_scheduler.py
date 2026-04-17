# масштабированное линейное расписание шума
# sqrt(beta)

import math

import torch


class ScaledLinearScheduler:

    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
    ) -> None:
        # дефолтные значения beta_min и beta_max - именно те, с которыми
        # обучался sdxl 
        self.beta_min = beta_min
        self.beta_max = beta_max
        # кэшируем квадратные корни - они используются в beta(t) на каждом
        # вызове, считать каждый раз заново нет смысла
        self.sqrt_beta_min = math.sqrt(beta_min)
        self.sqrt_beta_max = math.sqrt(beta_max)
        # размер дискретной сетки - тысяча шагов
        self.num_train_timesteps = num_train_timesteps

        # ленивые кэши под дискретные тензоры - считаются только при первом
        # обращении, дальше отдаются готовыми. самый дорогой - alphas_cumprod
        # (последовательное перемножение 1000 чисел)
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alphas_cumprod: torch.Tensor | None = None

    # beta в момент времени t [0,1].
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        sqrt_beta = self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min)
        return self.num_train_timesteps * sqrt_beta ** 2

    # накопленное значение α с начала до момента t - главная величина
    #
    # между точками заранее посчитанной дискретной таблицы. непрерывная
    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        # переводим t [0,1] в непрерывный индекс по массиву [0, N-1]
        idx = t * (self.num_train_timesteps - 1)
        idx = idx.clamp(0, self.num_train_timesteps - 1)

        # два соседних целых индекса и дробная часть между ними
        idx_low = idx.long()
        # clamp защищает от выхода за границу массива при t=1.0
        idx_high = (idx_low + 1).clamp(max=self.num_train_timesteps - 1)
        frac = idx - idx_low.float()

        ac = self.alphas_cumprod
        return ac[idx_low] * (1.0 - frac) + ac[idx_high] * frac

    # дискретный тензор из 1000 β, по одной на каждый шаг.
    # считается один раз при первом обращении, дальше кэш
    @property
    def betas(self) -> torch.Tensor:
        if self._betas is None:
            self._betas = self._compute_discrete_betas()
        return self._betas

    # α_t = 1 - β_t - доля сигнала, которая сохраняется за один шаг
    @property
    def alphas(self) -> torch.Tensor:
        if self._alphas is None:
            self._alphas = 1.0 - self.betas
        return self._alphas

    # ᾱ_t = α_1 · α_2 · ... · α_t - накопленное сохранение сигнала с начала.
    # именно эту таблицу использует alpha_bar(t) как опорные точки.
    # torch.cumprod делает накопленное произведение одной строкой
    @property
    def alphas_cumprod(self) -> torch.Tensor:
        if self._alphas_cumprod is None:
            self._alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        return self._alphas_cumprod

    # генерация дискретной таблицы β-шек под scaled linear:
    # 1000 равномерных точек в пространстве √β, потом возведение в квадрат
    def _compute_discrete_betas(self) -> torch.Tensor:
        sqrt_betas = torch.linspace(
            self.sqrt_beta_min,
            self.sqrt_beta_max,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        return sqrt_betas ** 2

    # σ(t) = √(1 - ᾱ(t)) - уровень шума в момент t.
    # при t≈0: σ≈0 (чистая картинка), при t=1: σ≈1 (чистый шум).
    # это тот самый σ, что фигурирует в шаге эйлера в солвере
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alpha_bar(t))

    # отношение сигнал/шум в момент t: ᾱ / (1-ᾱ).
    # при t=0 стремится к бесконечности (чистая картинка), при t=1 к нулю
    def snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab)

    # логарифм от snr - часто используется в теории диффузии как
    # "логарифмическое время". некоторые солверы удобнее формулировать
    # именно через log(snr), но в нашем основном цикле не используется
    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar(t)
        return torch.log(ab) - torch.log(1.0 - ab)
