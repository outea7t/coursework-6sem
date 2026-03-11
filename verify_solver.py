"""
Verification script: compare our DPM-Solver++ with diffusers' implementation.
Runs WITHOUT U-Net — uses synthetic model outputs.
"""

import torch
import numpy as np

def verify():
    print("=" * 60)
    print("DPM-Solver++ verification vs diffusers")
    print("=" * 60)

    # === 1. Compare schedule values ===
    from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler
    from src.sde.vp_sde import VPSDE
    from src.solvers.dpm_solver import DPMSolverPP

    scheduler = ScaledLinearScheduler()
    sde = VPSDE(scheduler=scheduler)
    solver = DPMSolverPP(sde, num_steps=30)

    print("\n--- Schedule values ---")
    ac = scheduler.alphas_cumprod
    print(f"alphas_cumprod[0]   = {ac[0]:.8f}")
    print(f"alphas_cumprod[499] = {ac[499]:.8f}")
    print(f"alphas_cumprod[999] = {ac[999]:.8f}")

    # Compare with diffusers
    try:
        from diffusers import DPMSolverMultistepScheduler
        diff_sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            solver_order=2,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
        )
        diff_ac = diff_sched.alphas_cumprod
        max_diff = (ac - diff_ac).abs().max().item()
        print(f"\nMax diff alphas_cumprod vs diffusers: {max_diff:.2e}")

        # Compare lambda values
        diff_alpha = torch.sqrt(diff_ac)
        diff_sigma = torch.sqrt(1 - diff_ac)
        diff_lambda = torch.log(diff_alpha / diff_sigma)
        our_lambda = solver._lambda_arr

        lambda_diff = (our_lambda - diff_lambda).abs().max().item()
        print(f"Max diff lambda vs diffusers: {lambda_diff:.2e}")

    except ImportError:
        print("diffusers not installed, skipping comparison")
        diff_sched = None

    # === 2. Compare timesteps ===
    print("\n--- Timesteps ---")
    print(f"Our timesteps (first 5):  {solver.timesteps[:5].tolist()}")
    print(f"Our timesteps (last 5):   {solver.timesteps[-5:].tolist()}")

    if diff_sched is not None:
        diff_sched.set_timesteps(30)
        diff_ts = diff_sched.timesteps.numpy()
        print(f"Diffusers timesteps (first 5): {diff_ts[:5].tolist()}")
        print(f"Diffusers timesteps (last 5):  {diff_ts[-5:].tolist()}")

        # Convert our continuous to discrete for comparison
        our_discrete = (solver.timesteps * 999).round().long().numpy()
        print(f"Our discrete (first 5):  {our_discrete[:5].tolist()}")
        print(f"Our discrete (last 5):   {our_discrete[-5:].tolist()}")

    # === 3. Single step comparison ===
    print("\n--- Single step comparison ---")
    torch.manual_seed(42)
    x = torch.randn(1, 4, 8, 8)
    eps = torch.randn(1, 4, 8, 8)

    # Our solver - first step
    solver.reset()
    t = solver.timesteps[0]
    t_prev = solver.timesteps[1]
    x_ours = solver.step(x.clone(), t, t_prev, eps.clone())

    print(f"t={t:.6f}, t_prev={t_prev:.6f}")
    print(f"x input  stats: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"x output stats: mean={x_ours.mean():.4f}, std={x_ours.std():.4f}")

    if diff_sched is not None:
        # Diffusers - first step
        diff_sched.set_timesteps(30)
        diff_ts_int = diff_sched.timesteps[0].item()

        # Convert epsilon to x0 (diffusers does this internally)
        diff_sigma = diff_sched.sigmas[0]
        diff_alpha_sigma = diff_sched._sigma_to_alpha_sigma_t(diff_sigma)
        alpha_s, sigma_s = diff_alpha_sigma

        x0_for_diff = (x - sigma_s * eps) / alpha_s

        # Diffusers step
        diff_sched._step_index = 0
        diff_sched.model_outputs = [None] * 2
        diff_out = diff_sched.step(eps.clone(), diff_ts_int, x.clone())
        x_diff = diff_out.prev_sample

        step_diff = (x_ours - x_diff).abs().max().item()
        print(f"\nDiffusers first step output: mean={x_diff.mean():.4f}, std={x_diff.std():.4f}")
        print(f"Max diff vs diffusers (step 0): {step_diff:.6f}")

    # === 4. Multi-step comparison (30 steps) ===
    if diff_sched is not None:
        print("\n--- 30-step comparison ---")
        torch.manual_seed(42)
        x_ours = torch.randn(1, 4, 8, 8)
        x_diff = x_ours.clone()

        solver.reset()
        diff_sched.set_timesteps(30)
        diff_sched.model_outputs = [None] * 2
        diff_sched._step_index = 0

        for i in range(30):
            # Use same random "model output" for both
            torch.manual_seed(100 + i)
            eps = torch.randn(1, 4, 8, 8)

            # Our solver
            t = solver.timesteps[i]
            t_prev = solver.timesteps[i + 1]
            x_ours = solver.step(x_ours.clone(), t, t_prev, eps.clone())

            # Diffusers solver
            diff_ts_int = diff_sched.timesteps[i].item()
            diff_sched._step_index = i
            diff_out = diff_sched.step(eps.clone(), diff_ts_int, x_diff.clone())
            x_diff = diff_out.prev_sample

            step_diff = (x_ours - x_diff).abs().max().item()
            if i < 3 or i >= 27 or step_diff > 0.1:
                print(f"Step {i:2d}: max_diff={step_diff:.6f}, "
                      f"ours mean={x_ours.mean():.4f}, "
                      f"diff mean={x_diff.mean():.4f}")

        final_diff = (x_ours - x_diff).abs().max().item()
        print(f"\nFinal max diff after 30 steps: {final_diff:.6f}")

    # === 5. Sanity check: solver reduces noise ===
    print("\n--- Sanity check: denoising ---")
    solver.reset()
    torch.manual_seed(42)
    x = torch.randn(1, 4, 8, 8)
    initial_std = x.std().item()

    for i in range(30):
        t = solver.timesteps[i]
        t_prev = solver.timesteps[i + 1]
        # Zero model output = "no noise predicted" = x0_pred should dominate
        eps = torch.zeros(1, 4, 8, 8)
        x = solver.step(x, t, t_prev, eps)

    print(f"Initial std: {initial_std:.4f}")
    print(f"Final std:   {x.std().item():.4f}")
    print(f"(Should decrease if solver is denoising correctly)")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    verify()
