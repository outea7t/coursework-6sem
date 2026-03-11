#!/usr/bin/env python3
"""
Bridge между Electron-приложением и диффузионным пайплайном.

Протокол: JSON lines через stdin/stdout.

Stdin (команды от Electron):
    {"type": "generate", "prompt": "...", "steps": 30, ...}

Stdout (события для Electron):
    {"type": "ready"}
    {"type": "generation_started", "total_steps": 30}
    {"type": "step", "step": 5, "total": 30, "image": "/path/to/step.png"}
    {"type": "generation_done", "image": "/path/to/final.png", "elapsed": 42.1}
    {"type": "error", "message": "..."}
"""

import json
import logging
import os
import sys
import tempfile
import time
import traceback

# Allow MPS to use all available memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Project root = directory containing this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Redirect all logging to stderr so stdout stays clean for JSON
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("bridge")


def send(data: dict) -> None:
    """Send a JSON message to Electron via stdout."""
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class LoadingProgressHandler(logging.Handler):
    """Intercept model loading logs and send progress events."""

    STAGES = {
        "Loading CLIP-L": (10, "Загрузка CLIP-L..."),
        "Loading OpenCLIP-G": (30, "Загрузка OpenCLIP-G..."),
        "Loading VAE": (55, "Загрузка VAE..."),
        "Loading U-Net": (75, "Загрузка U-Net..."),
        "All SDXL components loaded": (95, "Финализация..."),
    }

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        for pattern, (progress, label) in self.STAGES.items():
            if pattern in msg:
                send({"type": "loading_progress", "progress": progress, "message": label})
                break


def main() -> None:
    send({"type": "loading_progress", "progress": 0, "message": "Импорт библиотек..."})

    try:
        import torch
        torch.set_grad_enabled(False)
        from PIL import Image
        from src.pipeline.diffusion_pipeline import DiffusionPipeline
        from src.solvers import DPMSolverPP
        from src.guidance.cfg import ClassifierFreeGuidance
        from src.utils.device import randn_tensor
        from src.utils.seed import set_seed, get_generator

        # Attach progress handler to model loader
        progress_handler = LoadingProgressHandler()
        logging.getLogger("src.models.pretrained_loader").addHandler(progress_handler)

        send({"type": "loading_progress", "progress": 5, "message": "Инициализация пайплайна..."})

        pipeline = DiffusionPipeline(
            num_steps=30,
            guidance_scale=7.5,
        )

        logging.getLogger("src.models.pretrained_loader").removeHandler(progress_handler)

        send({"type": "loading_progress", "progress": 100, "message": "Готово"})
        send({"type": "ready"})
        logger.info("Pipeline ready, waiting for commands...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                send({"type": "error", "message": f"Invalid JSON: {line}"})
                continue

            if cmd.get("type") == "generate":
                prompt = cmd.get("prompt", "")
                negative_prompt = cmd.get(
                    "negative_prompt",
                    "low quality, blurry, distorted, ugly, bad anatomy",
                )
                steps = int(cmd.get("steps", 30))
                guidance = float(cmd.get("guidance", 7.5))
                seed = cmd.get("seed")
                if seed is not None:
                    seed = int(seed)
                width = int(cmd.get("width", 1024))
                height = int(cmd.get("height", 1024))

                # Update solver/cfg if params changed
                if steps != pipeline.num_steps:
                    pipeline.solver = DPMSolverPP(pipeline.sde, steps)
                    pipeline.num_steps = steps
                if guidance != pipeline.cfg.guidance_scale:
                    pipeline.cfg = ClassifierFreeGuidance(guidance)

                tmp_dir = tempfile.mkdtemp(prefix="diffusion_gen_")
                send({"type": "generation_started", "total_steps": steps})

                try:
                    start_time = time.time()

                    if seed is not None:
                        set_seed(seed)
                    generator = get_generator(seed, pipeline.device)

                    # Encode text
                    cond_embeds, cond_pooled = pipeline.models.encode_prompt(prompt)
                    uncond_embeds, uncond_pooled = pipeline.models.encode_prompt(
                        negative_prompt
                    )
                    time_ids = pipeline._build_time_ids(height, width)

                    # Initial noise
                    latent_shape = pipeline.model_config.get_latent_shape(height, width)
                    latents = randn_tensor(
                        latent_shape, pipeline.device, torch.float32, generator
                    )

                    # Setup solver
                    pipeline._setup_solver(
                        cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
                    )
                    timesteps = pipeline.solver.timesteps

                    total = len(timesteps) - 1
                    # Only 3 intermediate VAE decodes: ~33%, ~66%, last step
                    preview_at = set()
                    if total >= 3:
                        preview_at = {total // 3, 2 * total // 3, total - 1}
                    elif total >= 1:
                        preview_at = {total - 1}

                    is_mps = str(pipeline.device) == "mps"

                    for i in range(total):
                        t = timesteps[i].to(pipeline.device)
                        t_prev = timesteps[i + 1].to(pipeline.device)
                        discrete_t = pipeline._continuous_to_discrete(t)

                        noise_pred = pipeline._predict_noise(
                            latents,
                            discrete_t,
                            cond_embeds,
                            uncond_embeds,
                            cond_pooled,
                            uncond_pooled,
                            time_ids,
                        )

                        latents = pipeline.solver.step(
                            latents.float(), t, t_prev, noise_pred.float()
                        )

                        # Send progress every step (lightweight, no image)
                        step_num = i + 1
                        msg = {
                            "type": "progress",
                            "step": step_num,
                            "total": total,
                        }

                        # Decode preview at selected steps only
                        if step_num in preview_at:
                            if is_mps:
                                torch.mps.empty_cache()
                            img = pipeline._decode_and_postprocess(latents)
                            # Save as small JPEG thumbnail (~20KB vs ~4MB PNG)
                            thumb = img.resize((256, 256), Image.LANCZOS)
                            img_path = os.path.join(
                                tmp_dir, f"step_{step_num:04d}.jpg"
                            )
                            thumb.save(img_path, "JPEG", quality=75)
                            del img, thumb
                            if is_mps:
                                torch.mps.empty_cache()
                            msg["image"] = img_path

                        send(msg)

                    # Final image — full quality PNG
                    if is_mps:
                        torch.mps.empty_cache()
                    final_img = pipeline._decode_and_postprocess(latents)
                    final_path = os.path.join(tmp_dir, "final.png")
                    final_img.save(final_path)
                    del final_img
                    if is_mps:
                        torch.mps.empty_cache()

                    elapsed = time.time() - start_time
                    send(
                        {
                            "type": "generation_done",
                            "image": final_path,
                            "elapsed": round(elapsed, 1),
                        }
                    )
                    logger.info(f"Generation done in {elapsed:.1f}s")

                except Exception as e:
                    send({"type": "error", "message": str(e)})
                    traceback.print_exc(file=sys.stderr)

            elif cmd.get("type") == "ping":
                send({"type": "pong"})

    except Exception as e:
        send({"type": "error", "message": f"Fatal: {str(e)}"})
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
