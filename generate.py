#!/usr/bin/env python3
# запуск генерации изображения из командной строки
#
# пример: python3 generate.py "a majestic lion in the savannah"

import argparse
import logging
import os
import time


def main() -> None:
    # разбор аргументов командной строки.
    # для каждого параметра задаётся разумное значение по умолчанию,
    # чтобы можно было запустить просто с одним запросом
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

    # текстовый запрос - что нарисовать. единственный обязательный параметр
    parser.add_argument("prompt", type=str)
    # негативный запрос - что НЕ должно появиться на картинке.
    # помогает избавиться от типичных артефактов
    parser.add_argument("--negative_prompt", type=str,
        default="low quality, blurry, distorted, ugly, bad anatomy")
    # число шагов метода Эйлера. больше шагов - дольше генерация,
    # но картинка чуть детальнее. оптимум обычно 20..50
    parser.add_argument("--steps", type=int, default=30)
    # коэффициент усиления (см. cfg.py). больше - картинка ближе
    # к запросу, но при больших значениях появляются искажения
    parser.add_argument("--guidance", type=float, default=7.5)
    # зерно случайности. если не задано - выбирается случайно
    parser.add_argument("--seed", type=int, default=None)
    # размеры картинки. для sdxl оптимум 1024x1024
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # папка, куда сохранять готовую картинку
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--model", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0")
    # сохранять ли промежуточные шаги (полезно для демонстрации процесса)
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--intermediates_interval", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # настройка вывода логов
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # печатаем все параметры запуска - чтобы было видно,
    # что именно мы сейчас генерируем
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

    # если зерно не задано - берём случайное и сразу его печатаем,
    # чтобы можно было повторить результат, передав то же число
    if args.seed is None:
        import random
        args.seed = random.randint(0, 2**32 - 1)
        print(f"  Generated seed: {args.seed}")

    start_time = time.time()

    # тяжёлые библиотеки (torch, transformers, diffusers) импортируем
    # только сейчас - чтобы --help работал быстро, без ожидания
    from src.pipeline.diffusion_pipeline import DiffusionPipeline

    print("\nInitializing pipeline...")
    pipeline = DiffusionPipeline(
        model_id=args.model,
        device="auto",
        dtype="float16",
        num_steps=args.steps,
        guidance_scale=args.guidance,
    )

    # главный вызов - весь обратный процесс крутится внутри generate()
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

    # сохраняем готовую картинку. функция save_image сама собирает
    # имя файла из времени, запроса и параметров - чтобы файлы
    # не перезатирали друг друга
    from src.utils.image_utils import save_image
    filepath = save_image(
        image=image,
        output_dir=args.output,
        prompt=args.prompt,
        solver="euler",
        steps=args.steps,
        seed=args.seed,
    )

    # промежуточные шаги (если включены) кладём в отдельную папку.
    # имена с нулями слева - чтобы файлы сортировались по порядку шагов
    if intermediates:
        intermediates_dir = os.path.join(args.output, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)
        for i, img in enumerate(intermediates):
            int_path = os.path.join(
                intermediates_dir,
                f"step_{(i+1)*args.intermediates_interval:04d}.png",
            )
            img.save(int_path)
        print(f"\n  Saved {len(intermediates)} intermediate steps to {intermediates_dir}/")

    # сводка по итогам
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print(f"  Output:    {filepath}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Solver:    Euler ({args.steps} steps)")
    print("=" * 60)


if __name__ == "__main__":
    main()
