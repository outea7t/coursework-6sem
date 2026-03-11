"""
DPM-Solver++ — State-of-the-art солвер для диффузионных ODE.

DPM-Solver++ — специализированный солвер, разработанный именно для
диффузионных моделей. Использует экспоненциальный интегратор и
мультишаговую экстраполяцию для достижения высокого качества
генерации за 15-20 шагов.

Ключевая идея:
==============

    Вместо решения ODE в пространстве x(t), DPM-Solver работает
    в пространстве log-SNR (lambda). В этих координатах ODE имеет
    более простую структуру, что позволяет использовать специальные
    формулы интегрирования.

Математика:
===========

    Probability flow ODE для VP-SDE можно записать как:
        dx/dt = f(x,t) - 0.5*g^2*score

    Переход к координатам lambda(t) = log(alpha_bar(t)/(1-alpha_bar(t))):
        dx/dlambda = ... (упрощённая форма)

    Используя parametrization через x_0-prediction:
        x_t = alpha(t)*x_theta(x_t,t) + sigma(t)*eps_theta(x_t,t)

    DPM-Solver решает эту систему с помощью:
    1. Variation-of-constants формулы
    2. Экспоненциального интегратора
    3. Мультишаговой экстраполяции

DPM-Solver-1 (первого порядка):
    x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
              - alpha_{t-1} * (exp(-h) - 1) * eps_theta(x_t, t)

    где h = lambda_{t-1} - lambda_t (разность log-SNR)

    Это по сути аналог DDIM.

DPM-Solver-2 (второго порядка, мультишаговый):
    Использует предсказания шума на двух предыдущих шагах
    для экстраполяции, повышая точность до 2-го порядка.

    u = x_{t-1} (DPM-Solver-1)
    D1 = (eps_theta(x_t, t) - eps_theta(x_{t+1}, t+1)) / (lambda_t - lambda_{t+1})
    x_{t-1} = u - alpha_{t-1} * (exp(-h) - 1) / (2*lambda_diff) * h * D1

DPM-Solver++ (с dynamic thresholding):
    Добавляет thresholding предсказания x_0 для стабильности
    при высоких значениях guidance scale (>10).

Свойства:
    - Порядок: 1 или 2 (мультишаговый)
    - Детерминированный
    - 1 NFE на шаг (мультишаговый использует историю)
    - 15-20 шагов для хорошего качества (vs 50+ для Euler)
    - Оптимален для высокого CFG

Ссылки:
    [1] Lu, C., et al. (2022). "DPM-Solver: A Fast ODE Solver for
        Diffusion Probabilistic Model Sampling." (NeurIPS 2022).
    [2] Lu, C., et al. (2022). "DPM-Solver++: Fast Solver for Guided
        Sampling of Diffusion Probabilistic Models."
"""

import torch
from torch import Tensor

from .base_solver import BaseSolver
from ..sde.base_sde import BaseSDE


class DPMSolverPP(BaseSolver):
    """DPM-Solver++ солвер.

    Высокоэффективный солвер для диффузионных ODE.
    Использует мультишаговую экстраполяцию для 2-го порядка точности
    при 1 NFE на шаг (после первого шага).

    Args:
        sde: SDE модель.
        num_steps: Количество шагов.
        solver_order: Порядок солвера (1 или 2).
        thresholding: Применять dynamic thresholding для стабильности.
        dynamic_threshold_ratio: Квантиль для dynamic thresholding.
    """

    def __init__(
        self,
        sde: BaseSDE,
        num_steps: int = 20,
        solver_order: int = 2,
        thresholding: bool = False,
        dynamic_threshold_ratio: float = 0.995,
    ) -> None:
        super().__init__(sde=sde, num_steps=num_steps)
        self.solver_order = solver_order
        self.thresholding = thresholding
        self.dynamic_threshold_ratio = dynamic_threshold_ratio

        # История предыдущих x0-predictions для мультишаговой экстраполяции.
        # DPM-Solver++ работает в x0-пространстве (data prediction),
        # а не в epsilon-пространстве (noise prediction).
        self._x0_history: list[Tensor] = []
        self._lambda_history: list[Tensor] = []

        # Переопределяем timesteps: равномерная сетка в log-SNR пространстве
        # DPM-Solver++ оптимален с такой сеткой (шаги равномерны в lambda)
        self.timesteps = self._build_logsnr_timesteps()

    def _build_logsnr_timesteps(self) -> torch.Tensor:
        """Создаёт временную сетку, равномерную в log-SNR (lambda) пространстве.

        DPM-Solver++ работает в координатах lambda(t) = log(alpha_bar/(1-alpha_bar)).
        Равномерная сетка в lambda обеспечивает оптимальное распределение шагов.

        Returns:
            Тензор временных шагов (num_steps + 1,).
        """
        t_max = torch.tensor(self.sde.t_max)
        t_min = torch.tensor(self.sde.t_min)

        lambda_max = self.sde.scheduler.log_snr(t_min)  # высокий SNR (мало шума)
        lambda_min = self.sde.scheduler.log_snr(t_max)  # низкий SNR (много шума)

        # Равномерная сетка в lambda: от lambda_min (шум) к lambda_max (чисто)
        lambdas = torch.linspace(
            float(lambda_min), float(lambda_max), self.num_steps + 1
        )

        # Конвертация lambda -> t через бинарный поиск
        timesteps = []
        for lam in lambdas:
            t = self._lambda_to_t(float(lam))
            timesteps.append(t)

        return torch.tensor(timesteps, dtype=torch.float32)

    def _lambda_to_t(self, target_lambda: float) -> float:
        """Бинарный поиск t по заданному lambda (log-SNR).

        Args:
            target_lambda: Целевое значение log-SNR.

        Returns:
            Значение t, для которого log_snr(t) ≈ target_lambda.
        """
        lo, hi = float(self.sde.t_min), float(self.sde.t_max)
        for _ in range(64):  # 64 итерации дают точность ~1e-19
            mid = (lo + hi) / 2.0
            lam_mid = float(self.sde.scheduler.log_snr(torch.tensor(mid)))
            if lam_mid > target_lambda:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def reset(self) -> None:
        """Сброс истории (вызывать перед новой генерацией)."""
        self._x0_history.clear()
        self._lambda_history.clear()

    def step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        model_output: Tensor,
    ) -> Tensor:
        """Один шаг DPM-Solver++.

        Автоматически выбирает порядок:
        - Первый шаг: DPM-Solver-1 (нет истории)
        - Последующие шаги: DPM-Solver-2 (есть история)

        Args:
            x: Текущие латенты.
            t: Текущее время.
            t_prev: Следующее время.
            model_output: Предсказание шума epsilon.

        Returns:
            Латенты после одного шага.
        """
        # Вычисляем log-SNR (lambda) для текущего и следующего шагов
        lambda_t = self._log_snr(t)
        lambda_prev = self._log_snr(t_prev)
        h = lambda_prev - lambda_t  # > 0, так как lambda убывает по t

        # Коэффициенты alpha и sigma для текущего и следующего шагов
        alpha_t, sigma_t = self.sde.marginal_params_at_t(t)
        alpha_prev, sigma_prev = self.sde.marginal_params_at_t(t_prev)

        # Reshape для broadcasting
        for tensor_name in ['alpha_t', 'sigma_t', 'alpha_prev', 'sigma_prev']:
            tensor = locals()[tensor_name]
            while tensor.dim() < x.dim():
                tensor = tensor.unsqueeze(-1)
            locals()[tensor_name] = tensor
        # Re-assign after loop
        while alpha_t.dim() < x.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.dim() < x.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        while alpha_prev.dim() < x.dim():
            alpha_prev = alpha_prev.unsqueeze(-1)
        while sigma_prev.dim() < x.dim():
            sigma_prev = sigma_prev.unsqueeze(-1)

        # Конвертация noise prediction в x_0-prediction:
        # x_0 = (x_t - sigma_t * eps) / alpha_t
        x0_pred = (x - sigma_t * model_output) / alpha_t.clamp(min=1e-8)

        # Dynamic thresholding для стабильности при высоком CFG
        if self.thresholding:
            x0_pred = self._dynamic_threshold(x0_pred)

        # Сохраняем x0-prediction и lambda в историю
        self._x0_history.append(x0_pred.detach())
        self._lambda_history.append(lambda_t.detach())

        # Выбор порядка солвера
        if len(self._x0_history) < 2 or self.solver_order == 1:
            # DPM-Solver++ первого порядка (экспоненциальный интегратор)
            x_prev = self._dpm_solver_1(x, x0_pred, sigma_t, alpha_prev, sigma_prev, h)
        else:
            # DPM-Solver++ второго порядка (мультишаговая экстраполяция в x0-space)
            x_prev = self._dpm_solver_2(
                x, x0_pred, sigma_t,
                alpha_prev, sigma_prev, h
            )

        # Ограничиваем историю
        if len(self._x0_history) > 2:
            self._x0_history.pop(0)
            self._lambda_history.pop(0)

        return x_prev

    def _dpm_solver_1(
        self,
        x: Tensor,
        x0_pred: Tensor,
        sigma_t: Tensor,
        alpha_prev: Tensor,
        sigma_prev: Tensor,
        h: Tensor,
    ) -> Tensor:
        """DPM-Solver++ первого порядка (экспоненциальный интегратор).

        Формула (Lu et al., 2022, Algorithm 1):
            x_{t-1} = (sigma_{t-1}/sigma_t) * x_t
                      - alpha_{t-1} * (exp(-h) - 1) * D0

        где D0 = x0_pred (data prediction), h = lambda_{t-1} - lambda_t > 0.

        Знак "-" перед alpha критичен: так как exp(-h)-1 < 0,
        выражение -(exp(-h)-1) = (1-exp(-h)) > 0, и мы ДОБАВЛЯЕМ
        вклад x0_pred, сдвигая латенты к чистому изображению.

        Args:
            x: Текущие латенты.
            x0_pred: Предсказание x_0 (D0).
            sigma_t: sigma(t) (текущий шаг).
            alpha_prev: alpha(t_prev) (целевой шаг).
            sigma_prev: sigma(t_prev) (целевой шаг).
            h: Разность log-SNR (lambda_{t-1} - lambda_t), > 0.

        Returns:
            Латенты после шага.
        """
        return (sigma_prev / sigma_t) * x - alpha_prev * (torch.exp(-h) - 1) * x0_pred

    def _dpm_solver_2(
        self,
        x: Tensor,
        x0_pred: Tensor,
        sigma_t: Tensor,
        alpha_prev: Tensor,
        sigma_prev: Tensor,
        h: Tensor,
    ) -> Tensor:
        """DPM-Solver++ второго порядка (мультишаговый).

        Мультишаговая экстраполяция в x0-пространстве (data prediction).
        DPM-Solver++ использует именно x0-parametrization для стабильности
        при высоком CFG.

        Формула (Lu et al., 2022, Algorithm 2):
            D0 = x0_theta(x_t, t)  (текущая data prediction)
            r = h_{prev} / h  (отношение шагов log-SNR)
            D1 = (1/r) * (D0 - D0_prev)

            x_{t-1} = (sigma_{t-1}/sigma_t)*x_t
                      - alpha_{t-1}*(exp(-h)-1)*D0
                      - 0.5*alpha_{t-1}*(exp(-h)-1)*D1

        Args:
            x: Текущие латенты.
            x0_pred: Текущая data prediction (D0).
            sigma_t: sigma(t) (текущий шаг).
            alpha_prev: alpha(t_prev) (целевой шаг).
            sigma_prev: sigma(t_prev) (целевой шаг).
            h: Разность log-SNR (lambda_{t-1} - lambda_t), > 0.

        Returns:
            Латенты после шага.
        """
        # Текущая и предыдущая x0-predictions (data prediction space)
        D0 = x0_pred
        D0_prev = self._x0_history[-2]

        # Разность lambda предыдущего шага (h_prev)
        lambda_cur = self._lambda_history[-1]
        lambda_prev = self._lambda_history[-2]
        h_prev = lambda_cur - lambda_prev  # log-SNR разность предыдущего шага

        # Отношение шагов r = h_prev / h (как в diffusers)
        if abs(float(h)) > 1e-6:
            r = h_prev / h
        else:
            r = torch.ones_like(h)

        # D1 = (1/r) * (D0 - D0_prev) — включает масштабирование на h
        if abs(float(r)) > 1e-6:
            D1 = (1.0 / r) * (D0 - D0_prev)
        else:
            D1 = torch.zeros_like(D0)

        # Экспоненциальный интегратор (знак "-" критичен!)
        exp_neg_h = torch.exp(-h)
        x_prev = (sigma_prev / sigma_t) * x - alpha_prev * (exp_neg_h - 1) * D0

        # Коррекция второго порядка
        x_prev = x_prev - 0.5 * alpha_prev * (exp_neg_h - 1) * D1

        return x_prev

    def _log_snr(self, t: Tensor) -> Tensor:
        """Вычисляет log-SNR (lambda) для данного t.

        lambda(t) = log(alpha_bar(t) / (1 - alpha_bar(t)))

        Args:
            t: Время.

        Returns:
            log-SNR.
        """
        return self.sde.scheduler.log_snr(t)

    def _dynamic_threshold(self, x0_pred: Tensor) -> Tensor:
        """Dynamic thresholding для стабильности при высоком CFG.

        При больших guidance scale (>10) предсказание x_0 может выходить
        за допустимые пределы, вызывая артефакты. Dynamic thresholding
        масштабирует значения, чтобы квантиль не превышал 1.

        Алгоритм:
            1. Вычислить s = quantile(|x0_pred|, ratio) для каждого сэмпла
            2. Если s > 1: clamp |x0_pred| до s, затем делить на s

        Args:
            x0_pred: Предсказание x_0.

        Returns:
            x_0 после thresholding.
        """
        batch_size = x0_pred.shape[0]
        x0_flat = x0_pred.reshape(batch_size, -1)

        # Квантиль для каждого сэмпла в батче
        s = torch.quantile(
            x0_flat.abs().float(), self.dynamic_threshold_ratio, dim=1
        )
        # s должен быть >= 1
        s = s.clamp(min=1.0)

        # Reshape для broadcasting
        while s.dim() < x0_pred.dim():
            s = s.unsqueeze(-1)

        # Clamp и масштабирование
        x0_pred = x0_pred.clamp(-s, s) / s

        return x0_pred

    @property
    def is_stochastic(self) -> bool:
        return False

    @property
    def order(self) -> int:
        return self.solver_order

    @property
    def nfe_per_step(self) -> int:
        return 1  # Мультишаговый — использует историю
