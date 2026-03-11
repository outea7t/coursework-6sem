"""
Связь между предсказанием шума и score function.

Этот модуль демонстрирует математическую связь между различными
параметризациями нейронной сети в диффузионных моделях.

Основные параметризации:
========================

1. Epsilon-prediction (ε-prediction):
   Нейросеть предсказывает шум: eps_theta(x_t, t) ≈ eps
   где x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

   Используется в: DDPM, Stable Diffusion

2. Score-prediction (s-prediction):
   Нейросеть предсказывает score: s_theta(x_t, t) ≈ nabla_x log p_t(x_t)

   Используется в: NCSN, Score SDE

3. x_0-prediction:
   Нейросеть предсказывает x_0: x_theta(x_t, t) ≈ x_0

   Используется в: некоторых вариантах DDPM

4. v-prediction:
   Нейросеть предсказывает v: v_theta(x_t, t) ≈ v
   где v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0

   Используется в: Imagen, Progressive Distillation

Связи между параметризациями:
=============================

    Из q(x_t | x_0) = N(sqrt(alpha_bar)*x_0, (1-alpha_bar)*I):

    x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps

    Score function:
        nabla_x log q(x_t | x_0) = -(x_t - sqrt(alpha_bar)*x_0) / (1-alpha_bar)
                                   = -eps / sqrt(1-alpha_bar)
                                   = -eps / sigma(t)

    Следовательно:
        score = -eps / sigma(t)
        eps = -score * sigma(t)
        x_0 = (x_t - sigma(t)*eps) / sqrt(alpha_bar)
        v = sqrt(alpha_bar)*eps - sigma(t)*x_0

Ссылки:
    [1] Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models."
    [2] Song, Y., & Ermon, S. (2020). "Score-Based Generative Modeling."
    [3] Salimans, T., & Ho, J. (2022). "Progressive Distillation for
        Fast Sampling of Diffusion Models."
"""

import torch
from torch import Tensor


def eps_to_score(eps: Tensor, sigma_t: Tensor) -> Tensor:
    """Конвертация epsilon-prediction в score.

    score(x, t) = -eps / sigma(t)

    Вывод:
        nabla_x log q(x_t | x_0) = -(x_t - mu_t) / sigma_t^2
        Но x_t = mu_t + sigma_t * eps, поэтому x_t - mu_t = sigma_t * eps
        score = -sigma_t * eps / sigma_t^2 = -eps / sigma_t

    Args:
        eps: Предсказанный шум, (batch, C, H, W).
        sigma_t: sigma(t), скаляр или (batch,).

    Returns:
        Score function.
    """
    while sigma_t.dim() < eps.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    return -eps / sigma_t.clamp(min=1e-8)


def score_to_eps(score: Tensor, sigma_t: Tensor) -> Tensor:
    """Конвертация score в epsilon-prediction.

    eps = -score * sigma(t)

    Args:
        score: Score function.
        sigma_t: sigma(t).

    Returns:
        Epsilon (шум).
    """
    while sigma_t.dim() < score.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    return -score * sigma_t


def eps_to_x0(
    eps: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    """Конвертация epsilon-prediction в x_0-prediction.

    x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)

    Вывод:
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        => x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)

    Args:
        eps: Предсказанный шум.
        x_t: Зашумлённое состояние.
        alpha_bar_t: alpha_bar(t).

    Returns:
        Предсказание x_0.
    """
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar


def x0_to_eps(
    x_0: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    """Конвертация x_0-prediction в epsilon.

    eps = (x_t - sqrt(alpha_bar) * x_0) / sqrt(1 - alpha_bar)

    Args:
        x_0: Предсказание x_0.
        x_t: Зашумлённое состояние.
        alpha_bar_t: alpha_bar(t).

    Returns:
        Epsilon (шум).
    """
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar


def eps_to_v(
    eps: Tensor, x_0: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    """Конвертация epsilon и x_0 в v-prediction.

    v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0

    v-prediction объединяет информацию о шуме и сигнале.
    При высоком SNR (малый шум): v ≈ -x_0 (предсказываем сигнал)
    При низком SNR (много шума): v ≈ eps (предсказываем шум)

    Args:
        eps: Шум.
        x_0: Чистые данные.
        alpha_bar_t: alpha_bar(t).

    Returns:
        v-prediction.
    """
    while alpha_bar_t.dim() < eps.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x_0


def v_to_eps_x0(
    v: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> tuple[Tensor, Tensor]:
    """Конвертация v-prediction обратно в eps и x_0.

    Из системы:
        x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps
        v   = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x_0

    Решение:
        x_0 = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v
        eps = sqrt(1-alpha_bar)*x_t + sqrt(alpha_bar)*v

    Args:
        v: v-prediction.
        x_t: Зашумлённое состояние.
        alpha_bar_t: alpha_bar(t).

    Returns:
        (eps, x_0).
    """
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_ab = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_1mab = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))

    x_0 = sqrt_ab * x_t - sqrt_1mab * v
    eps = sqrt_1mab * x_t + sqrt_ab * v
    return eps, x_0
