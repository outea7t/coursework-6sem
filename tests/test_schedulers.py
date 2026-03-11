"""
Тесты для ScaledLinearScheduler (Stable Diffusion noise schedule).

Проверяют:
- Граничные условия: alpha_bar(0) ≈ 1, alpha_bar(1) < alpha_bar(0)
- Монотонность: alpha_bar убывает, beta > 0
- Корректность дискретных значений
"""

import torch
import pytest

from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler


@pytest.fixture
def scheduler():
    """ScaledLinearScheduler с параметрами по умолчанию."""
    return ScaledLinearScheduler()


class TestSchedulerBoundaryConditions:
    """Тесты граничных условий."""

    def test_alpha_bar_at_zero(self, scheduler):
        """alpha_bar(0) должно быть близко к 1 (чистый сигнал)."""
        t = torch.tensor(0.0)
        ab = scheduler.alpha_bar(t)
        assert ab > 0.95, f"alpha_bar(0) = {ab}, expected > 0.95"

    def test_alpha_bar_at_one(self, scheduler):
        """alpha_bar(1) должно быть < alpha_bar(0)."""
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


class TestScaledLinearScheduler:
    """Тесты для масштабированного линейного scheduler."""

    def test_default_params(self):
        """Параметры по умолчанию (Stable Diffusion)."""
        s = ScaledLinearScheduler()
        assert s.beta_min == 0.00085
        assert s.beta_max == 0.012
