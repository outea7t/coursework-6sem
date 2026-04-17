#!/usr/bin/env python3
# мост между electron и диффузионным пайплайном
# протокол: json lines через stdin/stdout

import json
import logging
import os
import sys
import tempfile
import time
import traceback

# снимаем лимит mps на высокую загрузку памяти - sdxl иначе упирается в потолок
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# bridge может быть запущен из директории app, добавляем корень проекта в пути
# чтобы работали импорты вида `from src.pipeline...`
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# логи в stderr - stdout зарезервирован под json-протокол с фронтом
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("bridge")


def send(data: dict) -> None:
    # отправка одного события клиенту: json-строка 
    # чтобы сообщение дошло немедленно, а не ждало буфера
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# перехватчик логов загрузки моделей: ищем в info-сообщениях
# знакомые паттерны и превращаем их в события прогресса для экрана загрузки
class LoadingProgressHandler(logging.Handler):

    # паттерн в логе -> (процент, подпись для ui)
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
    # первое событие для клиента - мы ещё не начали импортировать тяжёлые пакеты
    send({"type": "loading_progress", "progress": 0, "message": "Импорт библиотек..."})

    try:
        # стадия инициализации: импорты, создание пайплайна,
        # выход в состояние "готов принимать команды"

        import torch
        # мы только генерируем, не обучаем - отключаем подсчёт градиентов
        # глобально, иначе каждый вызов сети тянет за собой лишний граф вычислений
        torch.set_grad_enabled(False)
        from PIL import Image
        from src.pipeline.diffusion_pipeline import DiffusionPipeline
        from src.solvers import EulerSolver
        from src.guidance.cfg import ClassifierFreeGuidance
        from src.utils.device import randn_tensor
        from src.utils.seed import set_seed, get_generator

        # подписываем наш перехватчик логов на логгер загрузчика моделей -
        # теперь каждое info-сообщение оттуда будет проверяться на шаблоны
        # и при совпадении превращаться в событие прогресса для окна загрузки
        progress_handler = LoadingProgressHandler()
        logging.getLogger("src.models.pretrained_loader").addHandler(progress_handler)

        # сообщаем клиенту, что начинается тяжёлая часть - скачивание
        # и подгрузка весов sdxl в память видеокарты
        send({"type": "loading_progress", "progress": 5, "message": "Инициализация пайплайна..."})

        # собираем весь объект пайплайна за один вызов конструктора:
        pipeline = DiffusionPipeline(
            num_steps=30,
            guidance_scale=7.5,
        )

        # загрузка моделей завершена - снимаем перехватчик, чтобы он не
        # реагировал на возможные последующие логи
        logging.getLogger("src.models.pretrained_loader").removeHandler(progress_handler)

        # закрывающие события для окна загрузки: 100% и сигнал готовности.
        # получив "ready", клиент прячет экран загрузки и открывает окно чата
        send({"type": "loading_progress", "progress": 100, "message": "Готово"})
        send({"type": "ready"})
        logger.info("Pipeline ready, waiting for commands...")

        # стадия работы: бесконечно читаем команды из стандартного
        # ввода, каждая строка - отдельная команда в текстовом формате.
        # процесс живёт до тех пор, пока electron не закроет поток

        for line in sys.stdin:
            # убираем переносы строк и пробелы по краям
            line = line.strip()
            # пустые строки игнорируем (могут прилетать как разделители)
            if not line:
                continue

            # пытаемся разобрать строку как json-словарь с командой.
            # если прилетел битый текст - сообщаем фронту и ждём следующую,
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                send({"type": "error", "message": f"Invalid JSON: {line}"})
                continue

            # команда типа "generate" - пользователь нажал кнопку генерации
            if cmd.get("type") == "generate":
                # извлекаем параметры из полученной команды. у каждого поля
                # свой разумный дефолт на случай, если фронт его не передал
                prompt = cmd.get("prompt", "")                            # что рисуем
                negative_prompt = cmd.get(                                # чего избегать
                    "negative_prompt",
                    "low quality, blurry, distorted, ugly, bad anatomy",
                )
                steps = int(cmd.get("steps", 30))                         # число шагов солвера
                guidance = float(cmd.get("guidance", 7.5))                # сила влияния промпта
                seed = cmd.get("seed")                                    # зерно случайности
                if seed is not None:
                    seed = int(seed)
                width = int(cmd.get("width", 1024))                       # размер картинки
                height = int(cmd.get("height", 1024))

                # если пользователь в ui
                # поменял число шагов или силу подсказки между генерациями,
                # пересоздаём только те компоненты, которые от них зависят.
                # сами модели sdxl при этом остаются в памяти нетронутыми
                if steps != pipeline.num_steps:
                    pipeline.solver = EulerSolver(pipeline.sde, steps)
                    pipeline.num_steps = steps
                if guidance != pipeline.cfg.guidance_scale:
                    pipeline.cfg = ClassifierFreeGuidance(guidance)

                # под каждую генерацию выдаём свежую временную папку -
                # туда будут сохраняться превью-миниатюры с промежуточных
                # шагов и финальная полноразмерная картинка
                tmp_dir = tempfile.mkdtemp(prefix="diffusion_gen_")
                # сообщаем клиенту, что генерация начата и сколько всего
                # шагов ожидать - чтобы он мог нарисовать прогресс-бар
                send({"type": "generation_started", "total_steps": steps})

                # оборачиваем всю генерацию в try: любая ошибка внутри
                # одной картинки не убивает весь процесс
                try:
                    start_time = time.time()

                    if seed is not None:
                        set_seed(seed)
                    generator = get_generator(seed, pipeline.device)

                    # sdxl использует два текстовых энкодера одновременно:
                    # CLIP-L и OpenCLIP-G. их выходы склеиваются. на выходе
                    # получаем две штуки: последовательность векторов по
                    # токенам промпта и один сжатый вектор смысла целиком.
                    # оба нужны как условие для денойзинга
                    cond_embeds, cond_pooled = pipeline.models.encode_prompt(prompt)
                    # то же самое для негативного промпта.
                    uncond_embeds, uncond_pooled = pipeline.models.encode_prompt(
                        negative_prompt
                    )
                    # служебный тензор именно для sdxl: исходный размер
                    # картинки, координаты обрезки и целевой размер.
                    # unet от sdxl принимает его как дополнительный вход -
                    # модель училась учитывать эти размеры в качестве условия
                    time_ids = pipeline._build_time_ids(height, width)

                    # создание начального шума 
                    # модели sdxl работают не в пикселях, а в сжатом
                    # пространстве: в 8 раз меньше по каждой
                    # стороне и 4 канала вместо 3. для 1024x1024 картинки
                    # латент получается 128x128x4
                    latent_shape = pipeline.model_config.get_latent_shape(height, width)

                    latents = randn_tensor(
                        latent_shape, pipeline.device, torch.float32, generator
                    )

                    # подготовка к генерации
                    # сбрасываем внутреннее состояние на случай,
                    # если прошлый запуск что-то там оставил
                    pipeline._setup_solver(
                        cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
                    )
                    # заранее рассчитанная сетка моментов времени:
                    # от t=1 (чистый шум) до t≈0 (готовая картинка)
                    timesteps = pipeline.solver.timesteps

                    # сетка моментов времени для метода эйлера
                    total = len(timesteps) - 1
                    # выбор шагов для отправки превью-миниатюр 
                    preview_at = set()
                    if total >= 3:
                        preview_at = {total // 3, 2 * total // 3, total - 1}
                    elif total >= 1:
                        preview_at = {total - 1}

                    # флажок для устройств apple - нужен для ручной очистки
                    # памяти видеокарты перед декодированием
                    is_mps = str(pipeline.device) == "mps"

                    # главный цикл обратной диффузии.
                    # на каждой итерации делаем один шаг эйлера:
                    # берём текущий зашумлённый вариант, предсказываем
                    # сколько в нём шума, чуть-чуть его вычитаем
                    for i in range(total):
                        # текущий момент времени и следующий (ближе к 0)
                        t = timesteps[i].to(pipeline.device)
                        t_prev = timesteps[i + 1].to(pipeline.device)
                        # unet 
                        discrete_t = pipeline._continuous_to_discrete(t)

                        # два прохода нейросети для направляющей подсказки
                        # внутри _predict_noise сеть вызывается дважды:
                        # с условием (позитивный промпт) и без (негативный).
                        # результат экстраполируется по формуле:
                        # eps = eps_uncond + guidance * (eps_cond - eps_uncond)
                        # именно так промпт "сильнее" влияет на итог картинки
                        noise_pred = pipeline._predict_noise(
                            latents,
                            discrete_t,
                            cond_embeds,
                            uncond_embeds,
                            cond_pooled,
                            uncond_pooled,
                            time_ids,
                        )

                        # --- один шаг эйлера ---
                        # солвер берёт зашумлённый латент, предсказанный
                        # шум и переходит на следующий момент времени.
                        latents = pipeline.solver.step(
                            latents.float(), t, t_prev, noise_pred.float()
                        )

                        # --- сообщение клиенту о прогрессе ---
                        step_num = i + 1
                        msg = {
                            "type": "progress",
                            "step": step_num,
                            "total": total,
                        }

                        # если попали на контрольный шаг - декодируем латент
                        # в превью-миниатюру и кладём путь к ней в сообщение
                        if step_num in preview_at:
                            # перед вызовом vae чистим кэш видеокарты:
                            if is_mps:
                                torch.mps.empty_cache()
                            # vae декодирует латент в полноразмерную картинку
                            img = pipeline._decode_and_postprocess(latents)
                            # ужимаем до 256x256 - для предпросмотра
                            thumb = img.resize((256, 256), Image.LANCZOS)
                            img_path = os.path.join(
                                tmp_dir, f"step_{step_num:04d}.jpg"
                            )
                            # сохраняем в jpeg - для превью легче, чем png
                            thumb.save(img_path, "JPEG", quality=75)
                            del img, thumb
                            # ещё раз чистим память после декодирования
                            if is_mps:
                                torch.mps.empty_cache()
                            msg["image"] = img_path

                        send(msg)

                    # финальное декодирование: превращаем последний
                    # латент в полноразмерную картинку через vae
                    if is_mps:
                        torch.mps.empty_cache()
                    final_img = pipeline._decode_and_postprocess(latents)
                    # сохраняем в png без потерь - это финальный результат
                    final_path = os.path.join(tmp_dir, "final.png")
                    final_img.save(final_path)
                    del final_img
                    if is_mps:
                        torch.mps.empty_cache()

                    # сообщаем клиенту, что всё готово: путь к финальной
                    # картинке и сколько секунд ушло на генерацию
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
                    # ошибка одной генерации - сообщаем клиенту
                    send({"type": "error", "message": str(e)})
                    traceback.print_exc(file=sys.stderr)

    except Exception as e:
        send({"type": "error", "message": f"Fatal: {str(e)}"})
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
