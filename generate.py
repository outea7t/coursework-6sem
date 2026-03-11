#!/usr/bin/env python3
"""
Генерация изображений через авторский диффузионный пайплайн.

Использует предобученные компоненты SDXL (CLIP, VAE, U-Net) и
самостоятельно реализованные SDE, солверы и schedulers.

Примеры использования:
    python generate.py "a beautiful sunset over the ocean"
    python generate.py "a cat sitting on a windowsill" --steps 50 --solver dpm_solver_pp --guidance 9.0
    python generate.py "cyberpunk city at night" --solver runge_kutta --steps 100 --seed 42
    python generate.py "portrait of a wizard, fantasy art" --solver dpm_solver_pp --steps 20
    python generate.py "mountain landscape, photorealistic" --solver adaptive

Каждый вызов:
    1. Выводит информацию о параметрах (solver, steps, guidance, seed)
    2. Показывает прогресс-бар при генерации
    3. Сохраняет изображение в ./output/ с информативным именем файла
    4. Выводит путь к сохранённому файлу и время генерации
"""

import argparse
import logging
import os
import time

import yaml


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Загрузка конфигурации из YAML файла."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion Pipeline — Text-to-Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py "a majestic lion in the savannah, golden hour lighting"
  python generate.py "futuristic cityscape at night" --solver runge_kutta --steps 50
  python generate.py "a red rose with morning dew" --solver euler_maruyama --steps 50 --seed 42
  python generate.py "portrait of a wizard" --solver dpm_solver_pp --steps 20 --guidance 9.0
        """,
    )

    parser.add_argument(
        "prompt", type=str, help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted, ugly, bad anatomy",
        help="Negative prompt (default: standard quality filter)",
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of diffusion steps (default: 30)"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dpm_solver_pp",
        choices=[
            "euler_maruyama",
            "euler_ode",
            "runge_kutta",
            "heun",
            "dpm_solver_pp",
            "adaptive",
        ],
        help="Numerical solver for reverse SDE/ODE (default: dpm_solver_pp)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="scaled_linear",
        choices=["linear", "cosine", "scaled_linear", "continuous"],
        help="Noise schedule (default: scaled_linear)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="CFG guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save intermediate denoising steps",
    )
    parser.add_argument(
        "--intermediates_interval",
        type=int,
        default=5,
        help="Save intermediate every N steps (default: 5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Настройка логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Вывод параметров
    print("=" * 60)
    print("Diffusion Pipeline — Image Generation")
    print("=" * 60)
    print(f"  Prompt:    {args.prompt}")
    print(f"  Negative:  {args.negative_prompt}")
    print(f"  Solver:    {args.solver}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Steps:     {args.steps}")
    print(f"  Guidance:  {args.guidance}")
    print(f"  Size:      {args.width}x{args.height}")
    print(f"  Seed:      {args.seed if args.seed is not None else 'random'}")
    print(f"  Model:     {args.model}")
    print("=" * 60)

    # Генерация seed если не задан
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"  Generated seed: {args.seed}")

    start_time = time.time()

    # Инициализация пайплайна
    from src.pipeline.diffusion_pipeline import DiffusionPipeline

    print("\nInitializing pipeline...")
    pipeline = DiffusionPipeline(
        model_id=args.model,
        device="auto",
        dtype="float16",
        scheduler_name=args.scheduler,
        solver_name=args.solver,
        num_steps=args.steps,
        guidance_scale=args.guidance,
    )

    # Генерация
    print("\nGenerating image...")
    image, intermediates = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        save_intermediates=args.save_intermediates,
        intermediates_interval=args.intermediates_interval,
    )

    # Сохранение результата
    from src.utils.image_utils import save_image

    filepath = save_image(
        image=image,
        output_dir=args.output,
        prompt=args.prompt,
        solver=args.solver,
        steps=args.steps,
        seed=args.seed,
    )

    # Сохранение промежуточных шагов
    if intermediates:
        intermediates_dir = os.path.join(args.output, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)
        for i, img in enumerate(intermediates):
            int_path = os.path.join(intermediates_dir, f"step_{(i+1)*args.intermediates_interval:04d}.png")
            img.save(int_path)
        print(f"\n  Saved {len(intermediates)} intermediate steps to {intermediates_dir}/")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"  Output:    {filepath}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Solver:    {args.solver} ({args.steps} steps)")
    if hasattr(pipeline.solver, 'total_nfe'):
        print(f"  Total NFE: {pipeline.solver.total_nfe}")
    print("=" * 60)


if __name__ == "__main__":
    main()
