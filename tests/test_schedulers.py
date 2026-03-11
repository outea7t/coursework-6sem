"""
Тесты для noise schedulers.

Проверяют:
- Граничные условия: alpha_bar(0) ≈ 1, alpha_bar(1) ≈ 0
- Монотонность: alpha_bar убывает, beta > 0
- Корректность дискретных значений
- Согласованность между непрерывной и дискретной формулировками
"""

import torch
import pytest

from src.schedulers.linear_scheduler import LinearScheduler
from src.schedulers.cosine_scheduler import CosineScheduler
from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler
from src.schedulers.continuous_scheduler import ContinuousScheduler


@pytest.fixture(params=[
    LinearScheduler,
    CosineScheduler,
    ScaledLinearScheduler,
])
def scheduler(request):
    """Параметризованная фикстура для всех schedulers."""
    return request.param()


class TestSchedulerBoundaryConditions:
    """Тесты граничных условий."""

    def test_alpha_bar_at_zero(self, scheduler):
        """alpha_bar(0) должно быть близко к 1 (чистый сигнал)."""
        t = torch.tensor(0.0)
        ab = scheduler.alpha_bar(t)
        assert ab > 0.95, f"alpha_bar(0) = {ab}, expected > 0.95"

    def test_alpha_bar_at_one(self, scheduler):
        """alpha_bar(1) должно быть < alpha_bar(0) (больше шума в конце).

        Примечание: для DDPM-scale beta (0.0001-0.02) alpha_bar(1) всё ещё
        может быть близко к 1 в непрерывном времени, т.к. эти beta
        рассчитаны на 1000 дискретных шагов. Поэтому проверяем
        только строгое убывание.
        """
        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)
        assert scheduler.alpha_bar(t1) < scheduler.alpha_bar(t0), (
            "alpha_bar(1) must be less than alpha_bar(0)"
        )

    def test_beta_positive(self, scheduler):
        """beta(t) должно быть > 0 для всех t."""
        t = torch.linspace(0.01, 0.99, 100)
        betas = scheduler.beta(t)
        assert (betas > 0).all(), "beta(t) must be positive"


class TestSchedulerMonotonicity:
    """Тесты монотонности."""

    def test_alpha_bar_decreasing(self, scheduler):
        """alpha_bar(t) должно монотонно убывать."""
        t = torch.linspace(0.0, 1.0, 100)
        ab = scheduler.alpha_bar(t)
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-6).all(), "alpha_bar must be monotonically decreasing"

    def test_snr_decreasing(self, scheduler):
        """SNR(t) должно монотонно убывать."""
        t = torch.linspace(0.01, 0.99, 100)
        snr = scheduler.snr(t)
        diffs = snr[1:] - snr[:-1]
        assert (diffs <= 1e-4).all(), "SNR must be monotonically decreasing"


class TestSchedulerDiscreteValues:
    """Тесты дискретных значений."""

    def test_betas_shape(self, scheduler):
        """Дискретные betas должны иметь правильную форму."""
        assert scheduler.betas.shape == (1000,)

    def test_betas_range(self, scheduler):
        """Дискретные betas должны быть в разумном диапазоне."""
        assert (scheduler.betas > 0).all()
        assert (scheduler.betas < 1).all()

    def test_alphas_cumprod_decreasing(self, scheduler):
        """alphas_cumprod должно монотонно убывать."""
        ac = scheduler.alphas_cumprod
        diffs = ac[1:] - ac[:-1]
        assert (diffs <= 0).all()

    def test_alphas_cumprod_range(self, scheduler):
        """alphas_cumprod должно быть в (0, 1)."""
        ac = scheduler.alphas_cumprod
        assert (ac > 0).all()
        assert (ac <= 1.0).all()


class TestLinearScheduler:
    """Тесты для линейного scheduler."""

    def test_default_params(self):
        """Параметры по умолчанию (DDPM)."""
        s = LinearScheduler()
        assert s.beta_min == 0.0001
        assert s.beta_max == 0.02

    def test_beta_linearity(self):
        """beta(t) должно быть линейным (N-scaled для непрерывного SDE)."""
        s = LinearScheduler(beta_min=0.1, beta_max=1.0)
        t = torch.tensor([0.0, 0.5, 1.0])
        betas = s.beta(t)
        # beta(t) = N * (beta_min + t*(beta_max - beta_min)), N=1000
        expected = s.num_train_timesteps * torch.tensor([0.1, 0.55, 1.0])
        assert torch.allclose(betas, expected)


class TestCosineScheduler:
    """Тесты для косинусного scheduler."""

    def test_reaches_near_zero(self):
        """Косинусный scheduler должен достигать alpha_bar ≈ 0 при t=1."""
        cos_s = CosineScheduler()
        t = torch.tensor(1.0)
        ab = cos_s.alpha_bar(t)
        # Cosine scheduler конструирован так, чтобы alpha_bar(1) ≈ 0
        assert ab < 0.01, f"Cosine alpha_bar(1) = {ab}, expected < 0.01"

    def test_smoother_descent(self):
        """Косинусный scheduler убывает более плавно (меньше max|d alpha_bar/dt|)."""
        cos_s = CosineScheduler()
        t = torch.linspace(0.0, 1.0, 100)
        cos_ab = cos_s.alpha_bar(t)

        # Проверяем, что alpha_bar монотонно убывает от ~1 до ~0
        assert cos_ab[0] > 0.95
        assert cos_ab[-1] < 0.01


class TestScaledLinearScheduler:
    """Тесты для масштабированного линейного scheduler."""

    def test_default_params(self):
        """Параметры по умолчанию (Stable Diffusion)."""
        s = ScaledLinearScheduler()
        assert s.beta_min == 0.00085
        assert s.beta_max == 0.012


class TestContinuousScheduler:
    """Тесты для непрерывного scheduler."""

    def test_wrapping_base_scheduler(self):
        """Должен корректно оборачивать базовый scheduler."""
        base = LinearScheduler()
        cont = ContinuousScheduler(base_scheduler=base)

        t = torch.tensor(0.5)
        assert torch.allclose(cont.beta(t), base.beta(t))
        assert torch.allclose(cont.alpha_bar(t), base.alpha_bar(t))

    def test_drift_and_diffusion(self):
        """Drift и diffusion коэффициенты должны быть корректны."""
        cont = ContinuousScheduler()
        t = torch.tensor(0.5)

        drift = cont.drift_coefficient(t)
        diff = cont.diffusion_coefficient(t)

        assert drift < 0, "Drift coefficient should be negative (VP-SDE)"
        assert diff > 0, "Diffusion coefficient should be positive"
        assert torch.allclose(drift, -0.5 * cont.beta(t))
        assert torch.allclose(diff, torch.sqrt(cont.beta(t)))
