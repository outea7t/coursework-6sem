# конвертации между параметризациями: epsilon, score, x0, v

import torch
from torch import Tensor


def eps_to_score(eps: Tensor, sigma_t: Tensor) -> Tensor:
    # score = -eps / sigma(t)
    while sigma_t.dim() < eps.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    return -eps / sigma_t.clamp(min=1e-8)


def score_to_eps(score: Tensor, sigma_t: Tensor) -> Tensor:
    # eps = -score * sigma(t)
    while sigma_t.dim() < score.dim():
        sigma_t = sigma_t.unsqueeze(-1)
    return -score * sigma_t


def eps_to_x0(
    eps: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    # x0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return (x_t - sqrt_one_minus_alpha_bar * eps) / sqrt_alpha_bar


def x0_to_eps(
    x_0: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    # eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return (x_t - sqrt_alpha_bar * x_0) / sqrt_one_minus_alpha_bar


def eps_to_v(
    eps: Tensor, x_0: Tensor, alpha_bar_t: Tensor
) -> Tensor:
    # v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
    while alpha_bar_t.dim() < eps.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_one_minus_alpha_bar = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))
    return sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x_0


def v_to_eps_x0(
    v: Tensor, x_t: Tensor, alpha_bar_t: Tensor
) -> tuple[Tensor, Tensor]:
    # обратная конвертация из v в eps и x0
    while alpha_bar_t.dim() < x_t.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
    sqrt_ab = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
    sqrt_1mab = torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8))

    x_0 = sqrt_ab * x_t - sqrt_1mab * v
    eps = sqrt_1mab * x_t + sqrt_ab * v
    return eps, x_0
