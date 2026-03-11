"""
Тесты для SDE реализаций.

Проверяют:
- Корректность коэффициентов drift и diffusion
- Маргинальное распределение (mean и std)
- Согласованность прямого сэмплирования и SDE интегрирования
- Обратный drift и ODE drift
- Prior sampling
"""

import torch
import pytest

from src.sde.vp_sde import VPSDE
from src.sde.ve_sde import VESDE
from src.sde.sub_vp_sde import SubVPSDE
from src.schedulers.linear_scheduler import LinearScheduler
from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler


@pytest.fixture
def vp_sde():
    """VP-SDE с линейным scheduler."""
    return VPSDE(scheduler=LinearScheduler())


@pytest.fixture
def ve_sde():
    """VE-SDE."""
    return VESDE(scheduler=LinearScheduler())


@pytest.fixture
def sub_vp_sde():
    """Sub-VP SDE."""
    return SubVPSDE(scheduler=LinearScheduler())


class TestVPSDE:
    """Тесты для Variance Preserving SDE."""

    def test_drift_shape(self, vp_sde):
        """Drift должен иметь ту же форму, что и вход."""
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor(0.5)
        drift = vp_sde.drift(x, t)
        assert drift.shape == x.shape

    def test_drift_formula(self, vp_sde):
        """f(x, t) = -0.5 * beta(t) * x."""
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        drift = vp_sde.drift(x, t)
        beta_t = vp_sde.scheduler.beta(t)
        expected = -0.5 * beta_t * x
        assert torch.allclose(drift, expected, atol=1e-6)

    def test_diffusion_formula(self, vp_sde):
        """g(t) = sqrt(beta(t))."""
        t = torch.tensor(0.5)
        diff = vp_sde.diffusion(t)
        beta_t = vp_sde.scheduler.beta(t)
        expected = torch.sqrt(beta_t)
        assert torch.allclose(diff, expected, atol=1e-6)

    def test_marginal_params_at_zero(self, vp_sde):
        """При t ≈ 0: mean ≈ x_0, std ≈ 0."""
        x_0 = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.001)
        mean, std = vp_sde.marginal_params(x_0, t)
        assert torch.allclose(mean, x_0, atol=0.01)
        assert std.max() < 0.1

    def test_marginal_params_at_one(self, vp_sde):
        """При t ≈ 1: mean уменьшается, std увеличивается.

        Примечание: для DDPM-scale beta (0.0001-0.02) значения при t=1
        в непрерывном времени всё ещё далеки от предельных, т.к. эти
        beta рассчитаны на 1000 дискретных шагов. Проверяем тренд.
        """
        x_0 = torch.randn(1, 4, 8, 8)
        t_early = torch.tensor(0.01)
        t_late = torch.tensor(0.999)
        _, std_early = vp_sde.marginal_params(x_0, t_early)
        mean_late, std_late = vp_sde.marginal_params(x_0, t_late)
        # std должен расти с временем
        assert std_late.mean() > std_early.mean()
        # mean должен уменьшаться по абсолютному значению
        assert mean_late.abs().mean() < x_0.abs().mean()

    def test_forward_sampling_consistency(self, vp_sde):
        """Прямое сэмплирование: x_t = mean * x_0 + std * epsilon."""
        x_0 = torch.randn(100, 4, 8, 8)
        t = torch.tensor(0.5)
        eps = torch.randn_like(x_0)

        mean, std = vp_sde.marginal_params(x_0, t)
        x_t = mean + std * eps

        # Проверяем статистики x_t
        alpha_bar = vp_sde.scheduler.alpha_bar(t)
        expected_var = float(alpha_bar) * x_0.var() + float(1 - alpha_bar)
        actual_var = float(x_t.var())
        # Допуск из-за конечного размера выборки
        assert abs(actual_var - expected_var) < 0.5

    def test_prior_sampling(self, vp_sde):
        """Prior = N(0, I)."""
        samples = vp_sde.prior_sampling((10000,))
        assert abs(samples.mean()) < 0.1
        assert abs(samples.std() - 1.0) < 0.1

    def test_reverse_drift(self, vp_sde):
        """reverse_drift = f(x,t) - g(t)^2 * score."""
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        score = torch.randn_like(x)

        rev_drift = vp_sde.reverse_drift(x, t, score)
        f = vp_sde.drift(x, t)
        g = vp_sde.diffusion(t)
        expected = f - g ** 2 * score
        assert torch.allclose(rev_drift, expected, atol=1e-6)

    def test_reverse_ode_drift(self, vp_sde):
        """reverse_ode_drift = f(x,t) - 0.5 * g(t)^2 * score."""
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        score = torch.randn_like(x)

        ode_drift = vp_sde.reverse_ode_drift(x, t, score)
        f = vp_sde.drift(x, t)
        g = vp_sde.diffusion(t)
        expected = f - 0.5 * g ** 2 * score
        assert torch.allclose(ode_drift, expected, atol=1e-6)

    def test_noise_score_roundtrip(self, vp_sde):
        """noise -> score -> noise должно быть тождеством."""
        noise = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)

        score = vp_sde.noise_to_score(noise, t)
        noise_back = vp_sde.score_to_noise(score, t)
        assert torch.allclose(noise, noise_back, atol=1e-5)


class TestVESDE:
    """Тесты для Variance Exploding SDE."""

    def test_drift_is_zero(self, ve_sde):
        """VE-SDE имеет нулевой drift."""
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        drift = ve_sde.drift(x, t)
        assert torch.allclose(drift, torch.zeros_like(x))

    def test_marginal_mean_is_x0(self, ve_sde):
        """В VE-SDE mean = x_0 (без затухания)."""
        x_0 = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        mean, std = ve_sde.marginal_params(x_0, t)
        assert torch.allclose(mean, x_0)

    def test_sigma_monotonic(self, ve_sde):
        """sigma(t) должно монотонно расти."""
        t = torch.linspace(0.01, 0.99, 100)
        sigmas = ve_sde.sigma(t)
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs > 0).all()

    def test_prior_sampling_scale(self, ve_sde):
        """Prior ~ N(0, sigma_max^2)."""
        samples = ve_sde.prior_sampling((10000,))
        expected_std = ve_sde.sigma_max
        assert abs(samples.std() - expected_std) < expected_std * 0.2


class TestSubVPSDE:
    """Тесты для Sub-VP SDE."""

    def test_drift_same_as_vp(self, sub_vp_sde, vp_sde):
        """Sub-VP SDE имеет тот же drift, что и VP-SDE."""
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        assert torch.allclose(sub_vp_sde.drift(x, t), vp_sde.drift(x, t))

    def test_diffusion_less_than_vp(self, sub_vp_sde, vp_sde):
        """Diffusion Sub-VP <= diffusion VP."""
        t = torch.tensor(0.5)
        assert sub_vp_sde.diffusion(t) <= vp_sde.diffusion(t)

    def test_marginal_variance_differs_from_vp(self, sub_vp_sde, vp_sde):
        """Sub-VP и VP имеют разные дисперсии маргинального распределения.

        Для Sub-VP: std = sqrt(1 - alpha_bar^2)
        Для VP: std = sqrt(1 - alpha_bar)

        При alpha_bar close to 1 (малые beta):
        1 - alpha_bar^2 = (1-ab)(1+ab) > (1-ab), так как 1+ab > 1.
        Поэтому Sub-VP std > VP std в этом режиме.

        При alpha_bar close to 0 (большие beta):
        1 - alpha_bar^2 ≈ 1 ≈ 1 - alpha_bar.
        Обе дисперсии близки к 1.
        """
        x_0 = torch.randn(1, 4, 8, 8)
        t = torch.tensor(0.5)
        _, std_sub = sub_vp_sde.marginal_params(x_0, t)
        _, std_vp = vp_sde.marginal_params(x_0, t)
        # Проверяем, что значения различаются
        assert not torch.allclose(std_sub, std_vp, atol=1e-6), (
            "Sub-VP and VP should have different marginal std"
        )
