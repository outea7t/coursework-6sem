#!/usr/bin/env python3
# генерация изображений через cli

import argparse
import logging
import os
import time


def main() -> None:
    # флаги при запуске, задаем значения по умолчанию
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
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # настройка логирования
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
    print(f"  Solver:    Euler (1st order)")
    print(f"  Steps:     {args.steps}")
    print(f"  Guidance:  {args.guidance}")
    print(f"  Size:      {args.width}x{args.height}")
    print(f"  Seed:      {args.seed if args.seed is not None else 'random'}")
    print(f"  Model:     {args.model}")
    print("=" * 60)

    # если seed не задан - берём случайный и сразу печатаем,
    # чтобы можно было воспроизвести результат повторным запуском
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"  Generated seed: {args.seed}")

    start_time = time.time()

    # тяжёлый импорт (torch, transformers, diffusers) откладываем
    from src.pipeline.diffusion_pipeline import DiffusionPipeline

    print("\nInitializing pipeline...")
    pipeline = DiffusionPipeline(
        model_id=args.model,
        device="auto",
        dtype="float16",
        num_steps=args.steps,
        guidance_scale=args.guidance,
    )

    # основной вызов - весь обратный процесс и цикл эйлера крутятся внутри generate()
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
        solver="euler",
        steps=args.steps,
        seed=args.seed,
    )

    # промежуточные шаги (если запрошены) складываем отдельной папкой,
    # имена с нулями слева - чтобы файлы правильно сортировались по порядку шагов
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
    print(f"  Solver:    Euler ({args.steps} steps)")
    print("=" * 60)

if __name__ == "__main__":
    main()
