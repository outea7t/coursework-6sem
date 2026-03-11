"""
Тесты для DPM-Solver++.

Проверяют:
- Свойства солвера (порядок, детерминированность)
- Временную сетку (форма, монотонность)
- Сброс истории
- Dynamic thresholding
"""

import torch
import pytest

from src.sde.vp_sde import VPSDE
from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler
from src.solvers.dpm_solver import DPMSolverPP


class TestDPMSolverProperties:
    """Тесты свойств DPM-Solver++."""

    @pytest.fixture
    def sde(self):
        return VPSDE(scheduler=ScaledLinearScheduler())

    def test_default_order(self, sde):
        """DPM-Solver++ по умолчанию — 2-го порядка."""
        solver = DPMSolverPP(sde, 10)
        assert solver.order == 2

    def test_is_deterministic(self, sde):
        """DPM-Solver++ должен быть детерминированным."""
        solver = DPMSolverPP(sde, 10)
        assert not solver.is_stochastic

    def test_nfe_per_step(self, sde):
        """1 NFE на шаг (мультишаговый)."""
        solver = DPMSolverPP(sde, 10)
        assert solver.nfe_per_step == 1

    def test_timesteps_shape(self, sde):
        """Временная сетка должна иметь правильную форму."""
        solver = DPMSolverPP(sde, 30)
        assert solver.timesteps.shape == (31,)

    def test_timesteps_decreasing(self, sde):
        """Временные шаги должны убывать (от T к 0)."""
        solver = DPMSolverPP(sde, 30)
        diffs = solver.timesteps[1:] - solver.timesteps[:-1]
        assert (diffs < 0).all()

    def test_reset_clears_history(self, sde):
        """reset() должен очистить историю."""
        solver = DPMSolverPP(sde, 10)
        solver._x0_history.append(torch.randn(1, 4, 8, 8))
        solver.reset()
        assert len(solver._x0_history) == 0

    def test_dynamic_threshold(self, sde):
        """Dynamic thresholding должен ограничивать значения."""
        solver = DPMSolverPP(sde, 10, thresholding=True)
        x = torch.randn(1, 4, 8, 8) * 10  # Большие значения
        x_thresh = solver._dynamic_threshold(x)
        assert x_thresh.abs().max() <= 1.0 + 1e-6

    def test_first_order_mode(self, sde):
        """solver_order=1 должен использовать только первый порядок."""
        solver = DPMSolverPP(sde, 10, solver_order=1)
        assert solver.order == 1

    def test_step_produces_output(self, sde):
        """Шаг солвера должен вернуть тензор правильной формы."""
        solver = DPMSolverPP(sde, 30)
        solver.reset()
        x = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        t = solver.timesteps[0]
        t_prev = solver.timesteps[1]
        x_out = solver.step(x, t, t_prev, eps)
        assert x_out.shape == x.shape
        assert not torch.isnan(x_out).any()
        assert not torch.isinf(x_out).any()
