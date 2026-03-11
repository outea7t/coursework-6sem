#!/usr/bin/env python3
# генерация изображений через cli

import argparse
import logging
import os
import time

import yaml


def load_config(config_path: str = "config/default.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion Pipeline - Text-to-Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py "a majestic lion in the savannah, golden hour lighting"
  python generate.py "futuristic cityscape at night" --steps 30 --seed 42
  python generate.py "portrait of a wizard" --steps 20 --guidance 9.0
        """,
    )

    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative_prompt", type=str,
        default="low quality, blurry, distorted, ugly, bad anatomy")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--intermediates_interval", type=int, default=5)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Diffusion Pipeline")
    print("=" * 60)
    print(f"  Prompt:    {args.prompt}")
    print(f"  Negative:  {args.negative_prompt}")
    print(f"  Solver:    DPM-Solver++ (2nd order)")
    print(f"  Steps:     {args.steps}")
    print(f"  Guidance:  {args.guidance}")
    print(f"  Size:      {args.width}x{args.height}")
    print(f"  Seed:      {args.seed if args.seed is not None else 'random'}")
    print(f"  Model:     {args.model}")
    print("=" * 60)

    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"  Generated seed: {args.seed}")

    start_time = time.time()

    from src.pipeline.diffusion_pipeline import DiffusionPipeline

    print("\nInitializing pipeline...")
    pipeline = DiffusionPipeline(
        model_id=args.model,
        device="auto",
        dtype="float16",
        num_steps=args.steps,
        guidance_scale=args.guidance,
    )

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

    from src.utils.image_utils import save_image

    filepath = save_image(
        image=image,
        output_dir=args.output,
        prompt=args.prompt,
        solver="dpm_solver_pp",
        steps=args.steps,
        seed=args.seed,
    )

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
    print(f"  Solver:    DPM-Solver++ ({args.steps} steps)")
    if hasattr(pipeline.solver, 'total_nfe'):
        print(f"  Total NFE: {pipeline.solver.total_nfe}")
    print("=" * 60)


if __name__ == "__main__":
    main()
