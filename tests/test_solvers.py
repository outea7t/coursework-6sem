"""
Тесты для солверов.

Проверяют сходимость солверов на известных ODE, где аналитическое
решение известно. Это позволяет убедиться, что реализация корректна
без необходимости запускать полный пайплайн генерации.

Тестовое ODE:
    dy/dt = -y,  y(0) = 1
    Аналитическое решение: y(t) = e^{-t}

    Это простое линейное ODE, на котором можно проверить:
    - Корректность порядка сходимости (порядок p: ошибка ~ h^p)
    - Устойчивость при различных размерах шага
    - Корректность адаптивного шага
"""

import math

import torch
import pytest

from src.sde.vp_sde import VPSDE
from src.schedulers.linear_scheduler import LinearScheduler
from src.solvers.euler_ode import EulerODESolver
from src.solvers.euler_maruyama import EulerMaruyamaSolver
from src.solvers.heun import HeunSolver
from src.solvers.runge_kutta import RungeKutta4Solver
from src.solvers.dpm_solver import DPMSolverPP


def solve_simple_ode(solver_class, num_steps, t_start=1.0, t_end=0.0, y0=math.e):
    """Решает dy/dt = -y с y(1) = e, ожидая y(0) = 1.

    Мы моделируем это как "обратный процесс" от t=1 до t=0.

    Для тестирования солверов мы создаём мок-SDE, у которого
    reverse_ode_drift = -y (т.е. наше тестовое ODE).
    """

    class MockSDE:
        """Мок-SDE для тестирования солверов."""
        t_min = 1e-5
        t_max = 1.0

        class scheduler:
            @staticmethod
            def alpha_bar(t):
                # Для целей тестирования
                return torch.exp(-t)

            @staticmethod
            def log_snr(t):
                ab = torch.exp(-t)
                return torch.log(ab / (1 - ab).clamp(min=1e-8))

        def reverse_ode_drift(self, x, t, score):
            # Для тестового ODE dy/dt = -y:
            # Мы хотим, чтобы drift = -x
            return -x

        def reverse_drift(self, x, t, score):
            return -x

        def diffusion(self, t):
            return torch.tensor(0.0)

        def drift(self, x, t):
            return torch.zeros_like(x)

        def marginal_params_at_t(self, t):
            return torch.tensor(1.0), torch.tensor(0.1)

    sde = MockSDE()

    # Создаём солвер
    if solver_class == DPMSolverPP:
        solver = solver_class(sde, num_steps, solver_order=1)
    else:
        solver = solver_class(sde, num_steps)

    # Начальное условие
    y = torch.tensor([[y0]])

    # Интегрируем от t=1 до t=0
    timesteps = torch.linspace(1.0, 1e-5, num_steps + 1)

    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        # model_output не используется напрямую в нашем мок-SDE
        model_output = torch.zeros_like(y)
        y = solver.step(y, t, t_prev, model_output)

    return float(y.item())


class TestSolverConvergence:
    """Тесты сходимости солверов на простом ODE."""

    def test_euler_ode_convergence(self):
        """Euler ODE должен сходиться к точному решению при увеличении шагов."""
        errors = []
        for n_steps in [50, 100, 200]:
            result = solve_simple_ode(EulerODESolver, n_steps)
            # Точное решение: y(0) = e^0 = 1 (при y(1) = e^{-1+1} = e^0... )
            # Мы решаем dy/dt = -y от t=1 до t=0
            # y(t) = y(1) * e^{-(t-1)} = y(1) * e^{1-t}
            # y(0) = y(1) * e^1
            # Если y(1) = e, то y(0) = e * e = e^2...
            #
            # Проще: dy/dt = -y, y(1) = e → y(t) = e^{2-t} → y(0) = e^2
            # Нет, dy/dt = -y, y(1) = e → y(t) = e * e^{-(t-1)} = e^{2-t}
            # y(0) = e^2 ≈ 7.389
            exact = math.e ** 2
            error = abs(result - exact)
            errors.append(error)

        # Ошибка должна уменьшаться
        assert errors[-1] < errors[0], (
            f"Error should decrease with more steps: {errors}"
        )

    def test_euler_ode_first_order(self):
        """Euler ODE — метод 1-го порядка: удвоение шагов ~ уменьшает ошибку в 2 раза."""
        exact = math.e ** 2
        e1 = abs(solve_simple_ode(EulerODESolver, 100) - exact)
        e2 = abs(solve_simple_ode(EulerODESolver, 200) - exact)

        if e1 > 1e-6:  # Избегаем деления на 0
            ratio = e1 / e2
            # Для метода 1-го порядка: ratio ≈ 2
            assert ratio > 1.5, f"Expected ~2x error reduction, got {ratio:.2f}"


class TestSolverProperties:
    """Тесты свойств солверов."""

    @pytest.fixture
    def sde(self):
        return VPSDE(scheduler=LinearScheduler())

    def test_euler_ode_is_deterministic(self, sde):
        """Euler ODE должен быть детерминированным."""
        solver = EulerODESolver(sde, 10)
        assert not solver.is_stochastic

    def test_euler_maruyama_is_stochastic(self, sde):
        """Euler-Maruyama должен быть стохастическим."""
        solver = EulerMaruyamaSolver(sde, 10)
        assert solver.is_stochastic

    def test_heun_is_second_order(self, sde):
        """Heun должен иметь порядок 2."""
        solver = HeunSolver(sde, 10)
        assert solver.order == 2
        assert solver.nfe_per_step == 2

    def test_rk4_is_fourth_order(self, sde):
        """RK4 должен иметь порядок 4."""
        solver = RungeKutta4Solver(sde, 10)
        assert solver.order == 4
        assert solver.nfe_per_step == 4

    def test_dpm_solver_default_order(self, sde):
        """DPM-Solver++ по умолчанию — 2-го порядка."""
        solver = DPMSolverPP(sde, 10)
        assert solver.order == 2

    def test_timesteps_shape(self, sde):
        """Временная сетка должна иметь правильную форму."""
        solver = EulerODESolver(sde, 30)
        assert solver.timesteps.shape == (31,)

    def test_timesteps_decreasing(self, sde):
        """Временные шаги должны убывать (от T к 0)."""
        solver = EulerODESolver(sde, 30)
        diffs = solver.timesteps[1:] - solver.timesteps[:-1]
        assert (diffs < 0).all()


class TestDPMSolverPP:
    """Тесты для DPM-Solver++."""

    @pytest.fixture
    def sde(self):
        return VPSDE(scheduler=LinearScheduler())

    def test_reset_clears_history(self, sde):
        """reset() должен очистить историю."""
        solver = DPMSolverPP(sde, 10)
        solver._noise_history.append(torch.randn(1, 4, 8, 8))
        solver.reset()
        assert len(solver._noise_history) == 0

    def test_dynamic_threshold(self, sde):
        """Dynamic thresholding должен ограничивать значения."""
        solver = DPMSolverPP(sde, 10, thresholding=True)
        x = torch.randn(1, 4, 8, 8) * 10  # Большие значения
        x_thresh = solver._dynamic_threshold(x)
        # После thresholding значения должны быть ограничены
        assert x_thresh.abs().max() <= 1.0 + 1e-6
