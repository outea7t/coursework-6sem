#!/usr/bin/env python3
"""
Diagnostic script for debugging black images in the SDXL diffusion pipeline.

Tests each component in isolation:
1. VAE decoding (float16 vs float32, NaN/Inf checks)
2. Scheduler values (alpha_bar, marginal_params)
3. UNet forward pass outputs
4. Full pipeline flow (latent stats at each stage)
"""

import sys
import math
import torch
import numpy as np

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def tensor_stats(name, t):
    """Print detailed statistics for a tensor."""
    t_f = t.float()
    has_nan = torch.isnan(t_f).any().item()
    has_inf = torch.isinf(t_f).any().item()
    all_zero = (t_f == 0).all().item()
    pct_zero = ((t_f == 0).sum().item() / t_f.numel()) * 100
    print("  [{}]  shape={}  dtype={}".format(name, tuple(t.shape), t.dtype))
    print("    min={:.6g}  max={:.6g}  mean={:.6g}  std={:.6g}".format(
        t_f.min().item(), t_f.max().item(), t_f.mean().item(), t_f.std().item()))
    print("    NaN={}  Inf={}  allZero={}  %zero={:.1f}%".format(
        has_nan, has_inf, all_zero, pct_zero))
    if has_nan:
        nan_count = torch.isnan(t_f).sum().item()
        print("    *** NaN count: {} / {} ***".format(nan_count, t_f.numel()))
    if has_inf:
        inf_count = torch.isinf(t_f).sum().item()
        print("    *** Inf count: {} / {} ***".format(inf_count, t_f.numel()))
    return has_nan, has_inf, all_zero


def separator(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 1. VAE Decode Test
# ──────────────────────────────────────────────────────────────
def test_vae(device):
    separator("TEST 1: VAE Decode")
    from diffusers import AutoencoderKL

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scaling_factor = 0.13025

    # Use small latents (64x64 -> 512x512 image) to save memory
    latent_h, latent_w = 64, 64
    print("\nUsing latent size {}x{} (-> {}x{} image)".format(
        latent_h, latent_w, latent_h * 8, latent_w * 8))

    # ---- float16 VAE ----
    print("\n--- VAE in float16 ---")
    vae_f16 = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae_f16.requires_grad_(False)

    torch.manual_seed(42)
    rand_latents = torch.randn(1, 4, latent_h, latent_w, device="cpu").to(device)
    tensor_stats("random latents (input)", rand_latents)

    # Scale as the pipeline does: latents / scaling_factor
    scaled_latents_f16 = (rand_latents / scaling_factor).to(torch.float16)
    tensor_stats("scaled latents (fp16)", scaled_latents_f16)

    with torch.no_grad():
        decoded_f16 = vae_f16.decode(scaled_latents_f16).sample
    nan16, inf16, zero16 = tensor_stats("VAE decode output (fp16)", decoded_f16)

    del vae_f16
    if device == "mps":
        torch.mps.empty_cache()

    # ---- float32 VAE ----
    print("\n--- VAE in float32 ---")
    vae_f32 = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae_f32.requires_grad_(False)

    scaled_latents_f32 = (rand_latents.float() / scaling_factor)
    tensor_stats("scaled latents (fp32)", scaled_latents_f32)

    with torch.no_grad():
        decoded_f32 = vae_f32.decode(scaled_latents_f32).sample
    nan32, inf32, zero32 = tensor_stats("VAE decode output (fp32)", decoded_f32)

    # ---- Comparison ----
    print("\n--- VAE Diagnosis ---")
    if nan16 and not nan32:
        print("  >>> PROBLEM: float16 VAE produces NaN but float32 does not.")
        print("  >>> FIX: Decode VAE in float32 (or use upcast_vae).")
    elif zero16 and not zero32:
        print("  >>> PROBLEM: float16 VAE produces all-zero but float32 does not.")
    elif nan16 and nan32:
        print("  >>> PROBLEM: Both fp16 and fp32 VAE produce NaN. Input latents may be bad.")
    else:
        print("  VAE decoding looks healthy in both dtypes.")

    # ---- Test with zero latents (simulates what happens if solver goes to zero) ----
    print("\n--- VAE with zero latents ---")
    zero_latents = torch.zeros(1, 4, latent_h, latent_w, device=device, dtype=torch.float32)
    with torch.no_grad():
        decoded_zero = vae_f32.decode(zero_latents).sample
    tensor_stats("VAE decode(zeros) fp32", decoded_zero)

    del vae_f32
    if device == "mps":
        torch.mps.empty_cache()

    return nan16, inf16


# ──────────────────────────────────────────────────────────────
# 2. Scheduler / SDE Test
# ──────────────────────────────────────────────────────────────
def test_scheduler():
    separator("TEST 2: Scheduler & SDE Values")

    # Add project to path
    sys.path.insert(0, "/Users/outeast/Desktop/course-project")
    from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler
    from src.sde.vp_sde import VPSDE

    scheduler = ScaledLinearScheduler()
    sde = VPSDE(scheduler=scheduler)

    print("\n--- alpha_bar at various t values ---")
    header = "  {:>8} | {:>12} | {:>12} | {:>12} | {:>12}".format(
        "t", "alpha_bar", "sqrt(ab)", "sigma", "log_snr")
    print(header)
    print("  " + "-" * 64)

    t_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    problems = []
    for t_val in t_values:
        t = torch.tensor(t_val)
        ab = scheduler.alpha_bar(t).item()
        sqrt_ab = math.sqrt(max(ab, 0))
        sigma = math.sqrt(max(1 - ab, 0))
        log_snr = scheduler.log_snr(t).item()
        print("  {:>8.4f} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.4f}".format(
            t_val, ab, sqrt_ab, sigma, log_snr))

        if ab < 0 or ab > 1:
            problems.append("alpha_bar({})={} OUT OF [0,1]!".format(t_val, ab))
        if math.isnan(ab) or math.isinf(ab):
            problems.append("alpha_bar({}) is NaN/Inf!".format(t_val))

    print("\n--- marginal_params_at_t ---")
    for t_val in [0.001, 0.5, 1.0]:
        t = torch.tensor(t_val)
        mean_c, std_c = sde.marginal_params_at_t(t)
        print("  t={}: mean_coeff={:.6f}, std={:.6f}".format(
            t_val, mean_c.item(), std_c.item()))

    print("\n--- beta(t) values ---")
    for t_val in [0.0, 0.5, 1.0]:
        t = torch.tensor(t_val)
        b = scheduler.beta(t).item()
        print("  t={}: beta={:.6f}".format(t_val, b))

    print("\n--- Discrete betas (first 5, last 5) ---")
    betas = scheduler.betas
    print("  first 5: {}".format(betas[:5].tolist()))
    print("  last  5: {}".format(betas[-5:].tolist()))
    print("  min={:.8f}  max={:.8f}".format(betas.min().item(), betas.max().item()))

    # Check timestep grid
    from src.solvers.dpm_solver import DPMSolverPP
    solver = DPMSolverPP(sde, num_steps=30)
    ts = solver.timesteps
    print("\n--- Solver timesteps (first 5, last 5) ---")
    print("  first 5: {}".format(ts[:5].tolist()))
    print("  last  5: {}".format(ts[-5:].tolist()))
    print("  Total: {} points ({} steps)".format(len(ts), len(ts) - 1))

    # Check that h = lambda_prev - lambda_t is positive
    print("\n--- Checking log-SNR monotonicity (h > 0?) ---")
    h_problems = 0
    for i in range(len(ts) - 1):
        lam_t = scheduler.log_snr(ts[i]).item()
        lam_prev = scheduler.log_snr(ts[i + 1]).item()
        h = lam_prev - lam_t
        if i < 3 or i >= len(ts) - 3:
            print("  step {}: t={:.4f}->{:.4f}  lambda={:.4f}->{:.4f}  h={:.4f}".format(
                i, ts[i].item(), ts[i + 1].item(), lam_t, lam_prev, h))
        if h <= 0:
            h_problems += 1
    if h_problems:
        print("  >>> PROBLEM: {} steps have h <= 0 (log-SNR not increasing)".format(h_problems))
    else:
        print("  All {} steps have h > 0. Good.".format(len(ts) - 1))

    if problems:
        print("\n  >>> SCHEDULER PROBLEMS:")
        for p in problems:
            print("    - " + p)
    else:
        print("\n  Scheduler values look correct.")


# ──────────────────────────────────────────────────────────────
# 3. UNet Forward Pass Test
# ──────────────────────────────────────────────────────────────
def test_unet(device):
    separator("TEST 3: UNet Forward Pass")
    from diffusers import UNet2DConditionModel

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float16

    print("\nLoading UNet ({})...".format(dtype))
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.requires_grad_(False)

    # Use small latents for speed (64x64)
    latent_h, latent_w = 64, 64

    torch.manual_seed(42)
    latents = torch.randn(1, 4, latent_h, latent_w, device="cpu", dtype=torch.float32).to(device).to(dtype)
    tensor_stats("UNet input latents", latents)

    # Create dummy text embeddings
    encoder_hidden_states = torch.randn(1, 77, 2048, device=device, dtype=dtype)
    text_embeds = torch.randn(1, 1280, device=device, dtype=dtype)
    time_ids = torch.tensor([[512., 512., 0., 0., 512., 512.]], device=device, dtype=dtype)

    # Test at different timesteps
    for discrete_t_val in [999, 500, 100, 1]:
        timestep = torch.tensor([discrete_t_val], device=device)
        with torch.no_grad():
            output = unet(
                latents,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
        has_nan, has_inf, _ = tensor_stats("UNet output (t={})".format(discrete_t_val), output)
        if has_nan:
            print("  >>> PROBLEM: UNet output has NaN at timestep {}".format(discrete_t_val))
        if has_inf:
            print("  >>> PROBLEM: UNet output has Inf at timestep {}".format(discrete_t_val))

    del unet
    if device == "mps":
        torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────────
# 4. Full Pipeline Flow (Simulated)
# ──────────────────────────────────────────────────────────────
def test_full_pipeline(device):
    separator("TEST 4: Full Pipeline Denoising Loop")
    sys.path.insert(0, "/Users/outeast/Desktop/course-project")

    from src.schedulers.scaled_linear_scheduler import ScaledLinearScheduler
    from src.sde.vp_sde import VPSDE
    from src.solvers.dpm_solver import DPMSolverPP
    from src.guidance.cfg import ClassifierFreeGuidance
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float16
    scaling_factor = 0.13025
    num_steps = 20  # Fewer steps for debugging speed
    guidance_scale = 7.5

    # Use 512x512 for faster debugging
    height, width = 512, 512
    latent_h, latent_w = height // 8, width // 8

    print("\nImage: {}x{}, Latents: {}x{}".format(height, width, latent_h, latent_w))
    print("Steps: {}, Guidance: {}".format(num_steps, guidance_scale))
    print("Device: {}, dtype: {}".format(device, dtype))

    # Load models
    print("\nLoading models...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.requires_grad_(False)

    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    text_encoder_2.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.requires_grad_(False)

    # Load VAE in float32 for decoding
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)

    # Encode prompt
    prompt = "a photo of an astronaut riding a horse on mars"
    negative_prompt = ""

    print("\nEncoding prompt: '{}'".format(prompt))

    def encode_prompt(text):
        tokens_1 = tokenizer(text, return_tensors="pt", padding="max_length",
                             max_length=77, truncation=True)
        with torch.no_grad():
            output_1 = text_encoder(tokens_1.input_ids.to(device))
        hs1 = output_1.last_hidden_state

        tokens_2 = tokenizer_2(text, return_tensors="pt", padding="max_length",
                               max_length=77, truncation=True)
        with torch.no_grad():
            output_2 = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
        hs2 = output_2.hidden_states[-2]
        pooled = output_2.text_embeds

        prompt_embeds = torch.cat([hs1, hs2], dim=-1)
        return prompt_embeds, pooled

    cond_embeds, cond_pooled = encode_prompt(prompt)
    uncond_embeds, uncond_pooled = encode_prompt(negative_prompt)

    tensor_stats("cond_embeds", cond_embeds)
    tensor_stats("cond_pooled", cond_pooled)
    tensor_stats("uncond_embeds", uncond_embeds)
    tensor_stats("uncond_pooled", uncond_pooled)

    # Setup scheduler, SDE, solver
    scheduler = ScaledLinearScheduler()
    sde = VPSDE(scheduler=scheduler)
    solver = DPMSolverPP(sde, num_steps=num_steps)
    cfg = ClassifierFreeGuidance(guidance_scale)

    time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype, device=device)

    # Initial latents
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    latents = torch.randn(1, 4, latent_h, latent_w, generator=gen, device="cpu", dtype=torch.float32).to(device)
    tensor_stats("initial latents (x_T)", latents)

    # Denoising loop
    timesteps = solver.timesteps
    print("\nStarting denoising: {} steps".format(len(timesteps) - 1))
    print("Timestep range: {:.4f} -> {:.4f}".format(timesteps[0].item(), timesteps[-1].item()))

    first_problem_step = None
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]

        # Convert continuous t to discrete timestep (as pipeline does)
        num_train = 1000
        discrete_t = (t * (num_train - 1)).long().clamp(0, num_train - 1).to(device)

        # CFG: predict noise
        with torch.no_grad():
            noise_pred = cfg(
                unet,
                latents.to(dtype),
                discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )
        noise_pred_f32 = noise_pred.float()

        # Solver step (in float32)
        latents = solver.step(latents.float(), t.to(device), t_prev.to(device), noise_pred_f32)

        # Check stats at key steps
        is_key_step = (i < 3) or (i >= len(timesteps) - 4) or (i % 5 == 0)
        has_nan = torch.isnan(latents).any().item()
        has_inf = torch.isinf(latents).any().item()
        all_zero = (latents == 0).all().item()

        if is_key_step or has_nan or has_inf or all_zero:
            print("\n  Step {}: t={:.4f} -> t_prev={:.4f} (discrete={})".format(
                i, t.item(), t_prev.item(), discrete_t.item()))
            tensor_stats("noise_pred (step {})".format(i), noise_pred_f32)
            tensor_stats("latents (step {})".format(i), latents)

            # Additional diagnostics: check marginal params
            alpha_t, sigma_t = sde.marginal_params_at_t(t)
            alpha_prev, sigma_prev = sde.marginal_params_at_t(t_prev)
            print("    alpha(t)={:.6f}  sigma(t)={:.6f}".format(alpha_t.item(), sigma_t.item()))
            print("    alpha(t_prev)={:.6f}  sigma(t_prev)={:.6f}".format(
                alpha_prev.item(), sigma_prev.item()))

        if (has_nan or has_inf or all_zero) and first_problem_step is None:
            first_problem_step = i
            print("  >>> FIRST PROBLEM at step {}!".format(i))
            # Print detailed debug for the step that goes bad
            print("  >>> Debugging solver.step internals at this step...")

            # Reproduce what DPMSolverPP.step does
            lambda_t_val = scheduler.log_snr(t)
            lambda_prev_val = scheduler.log_snr(t_prev)
            h_val = lambda_prev_val - lambda_t_val
            alpha_t_v, sigma_t_v = sde.marginal_params_at_t(t)
            alpha_prev_v, sigma_prev_v = sde.marginal_params_at_t(t_prev)
            print("    lambda_t={:.6f}  lambda_prev={:.6f}  h={:.6f}".format(
                lambda_t_val.item(), lambda_prev_val.item(), h_val.item()))

            # x0 prediction
            s_t = sigma_t_v
            a_t = alpha_t_v
            while s_t.dim() < latents.dim():
                s_t = s_t.unsqueeze(-1)
            while a_t.dim() < latents.dim():
                a_t = a_t.unsqueeze(-1)
            x0_pred = (latents.float() - s_t * noise_pred_f32) / a_t.clamp(min=1e-8)
            tensor_stats("  x0_pred (inside solver)", x0_pred)

    # Final latent stats
    separator("FINAL RESULTS")
    print("\n--- Final latents (after denoising) ---")
    tensor_stats("final latents", latents)

    # Decode with float32 VAE
    print("\n--- Decoding with float32 VAE ---")
    latents_for_decode = latents.float() / scaling_factor
    tensor_stats("latents / scaling_factor", latents_for_decode)

    with torch.no_grad():
        image_tensor = vae.decode(latents_for_decode).sample
    tensor_stats("VAE decoded image tensor (fp32)", image_tensor)

    # Also try decoding with fp16 (as the pipeline does by default)
    print("\n--- Decoding with float16 VAE (as pipeline does) ---")
    del vae
    if device == "mps":
        torch.mps.empty_cache()

    vae_f16 = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae_f16.requires_grad_(False)

    latents_for_decode_f16 = (latents / scaling_factor).to(torch.float16)
    tensor_stats("latents / scaling_factor (fp16)", latents_for_decode_f16)

    with torch.no_grad():
        image_tensor_f16 = vae_f16.decode(latents_for_decode_f16).sample
    tensor_stats("VAE decoded image tensor (fp16)", image_tensor_f16)

    # Post-processing
    print("\n--- Post-processing (tensor_to_pil equivalent) ---")
    # Use fp32 decoded image
    img_t = image_tensor.float().clamp(-1, 1)
    img_01 = (img_t.squeeze(0) + 1.0) / 2.0
    tensor_stats("image [0,1]", img_01)
    img_uint8 = (img_01.permute(1, 2, 0).cpu().numpy() * 255).round()
    print("  uint8 range: min={:.0f}  max={:.0f}  mean={:.1f}".format(
        img_uint8.min(), img_uint8.max(), img_uint8.mean()))

    if img_uint8.max() < 5:
        print("  >>> PROBLEM: Final image is essentially BLACK (max pixel < 5)")
    elif img_uint8.max() < 30:
        print("  >>> PROBLEM: Final image is very dark (max pixel < 30)")
    else:
        print("  Image pixel range looks reasonable.")

    # fp16 decoded version
    print("\n--- Post-processing fp16 VAE output ---")
    img_t_f16 = image_tensor_f16.float().clamp(-1, 1)
    img_01_f16 = (img_t_f16.squeeze(0) + 1.0) / 2.0
    tensor_stats("image [0,1] (fp16 VAE)", img_01_f16)
    img_uint8_f16 = (img_01_f16.permute(1, 2, 0).cpu().numpy() * 255).round()
    print("  uint8 range: min={:.0f}  max={:.0f}  mean={:.1f}".format(
        img_uint8_f16.min(), img_uint8_f16.max(), img_uint8_f16.mean()))

    if img_uint8_f16.max() < 5:
        print("  >>> PROBLEM: fp16 VAE produces BLACK image (max pixel < 5)")

    # Cleanup
    del unet, vae_f16, text_encoder, text_encoder_2
    if device == "mps":
        torch.mps.empty_cache()

    if first_problem_step is not None:
        print("\n  >>> First numerical problem appeared at step {}".format(first_problem_step))
    else:
        print("\n  No NaN/Inf/zero issues detected in the denoising loop.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("PyTorch version: {}".format(torch.__version__))
    print("Device: {}".format(device))
    print("MPS available: {}".format(torch.backends.mps.is_available()))

    # Run scheduler test first (no GPU needed, fast)
    test_scheduler()

    # VAE test
    vae_nan, vae_inf = test_vae(device)

    # UNet test
    test_unet(device)

    # Full pipeline
    test_full_pipeline(device)

    separator("SUMMARY")
    print("\nDiagnostic complete. Look for '>>> PROBLEM' markers above.")
    print("Common causes of black images:")
    print("  1. VAE NaN in float16 -> decode in float32")
    print("  2. Wrong scaling_factor or missing division by scaling_factor")
    print("  3. Solver producing NaN/Inf latents (bad scheduler params)")
    print("  4. Dynamic thresholding clamping x0_pred to near-zero")
    print("  5. Timestep mapping issues (continuous t -> discrete)")
    print("  6. Latents collapsing to zero during denoising (wrong SDE coefficients)")
