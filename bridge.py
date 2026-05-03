#!/usr/bin/env python3
# мост между десктопным приложением (Electron) и диффузионным пайплайном
#
# как это работает:
#   - electron-приложение запускает этот скрипт как дочерний процесс;
#   - скрипт читает команды из стандартного ввода (по одной json-строке);
#   - в ответ пишет в стандартный вывод события (тоже json-строки):
#     прогресс загрузки, прогресс генерации, превью-миниатюры, ошибки.
# то есть это простая текстовая связь "запрос-событие".

import json
import logging
import os
import sys
import tempfile
import time
import traceback

# на видеокарте apple sdxl упирается в потолок памяти.
# эта переменная окружения снимает ограничение - mps будет
# использовать столько памяти, сколько надо
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# на windows stdout может использовать системную кодировку (cp1251),
# а нам нужен utf-8 для корректной передачи кириллицы в electron
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# bridge может быть запущен из директории app, поэтому добавляем
# корень проекта в пути поиска модулей. иначе импорты
# вида `from src.pipeline...` не сработают
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# логи отправляем в stderr - stdout зарезервирован под общение с приложением
# (через json-протокол). если смешать - приложение не сможет разобрать ответ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("bridge")


def send(data: dict) -> None:
    # отправка одного события клиенту:
    # сериализуем словарь в json, добавляем перенос строки и
    # сразу сбрасываем буфер - чтобы приложение получило сообщение
    # сразу, а не ждало пока заполнится буфер вывода
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class LoadingProgressHandler(logging.Handler):
    # перехватчик логов загрузки моделей.
    # ищем в info-сообщениях знакомые паттерны и превращаем
    # их в события прогресса для экрана загрузки приложения

    # шаблон в логе -> (процент, подпись для интерфейса)
    STAGES = {
        "Loading CLIP-L": (10, "Загрузка CLIP-L... (~250 МБ)"),
        "Loading OpenCLIP-G": (30, "Загрузка OpenCLIP-G... (~1.4 ГБ)"),
        "Loading VAE": (55, "Загрузка VAE... (~160 МБ)"),
        "Loading U-Net": (75, "Загрузка U-Net... (~5.1 ГБ)"),
        "All SDXL components loaded": (95, "Финализация..."),
    }

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        # перебираем шаблоны и при совпадении отправляем событие
        for pattern, (progress, label) in self.STAGES.items():
            if pattern in msg:
                send({"type": "loading_progress", "progress": progress, "message": label})
                break


def main() -> None:
    # первое событие - чтобы окно загрузки показало хоть что-то,
    # пока импорты не завершатся
    send({"type": "loading_progress", "progress": 0, "message": "Импорт библиотек..."})

    try:
        # === стадия 1: инициализация ===

        import torch
        # глобально отключаем подсчёт градиентов: мы не обучаем сеть.
        # без этого каждый вызов сети тянул бы лишний граф и память
        torch.set_grad_enabled(False)
        from PIL import Image
        from src.pipeline.diffusion_pipeline import DiffusionPipeline
        from src.solvers import EulerSolver
        from src.guidance.cfg import ClassifierFreeGuidance
        from src.utils.device import randn_tensor
        from src.utils.seed import set_seed, get_generator

        # подписываемся на логи загрузчика моделей - чтобы каждое
        # info-сообщение от него превращалось в событие прогресса
        progress_handler = LoadingProgressHandler()
        logging.getLogger("src.models.pretrained_loader").addHandler(progress_handler)

        # начинается тяжёлая часть - скачивание/загрузка весов sdxl в память
        send({"type": "loading_progress", "progress": 5, "message": "Инициализация пайплайна..."})

        # собираем весь пайплайн за один вызов конструктора
        pipeline = DiffusionPipeline(
            num_steps=30,
            guidance_scale=7.5,
        )

        # снимаем перехватчик - дальше он не нужен
        logging.getLogger("src.models.pretrained_loader").removeHandler(progress_handler)

        # сообщаем приложению, что всё готово.
        # получив "ready", приложение прячет окно загрузки и открывает чат
        send({"type": "loading_progress", "progress": 100, "message": "Готово"})
        send({"type": "ready"})
        logger.info("Pipeline ready, waiting for commands...")

        # === стадия 2: цикл команд ===
        # читаем команды из стандартного ввода. процесс живёт,
        # пока приложение не закроет поток ввода

        for line in sys.stdin:
            # убираем переносы строк и пробелы по краям
            line = line.strip()
            # пустые строки игнорируем
            if not line:
                continue

            # пытаемся разобрать строку как json. если прилетел
            # некорректный текст - сообщаем приложению и ждём следующую команду
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                send({"type": "error", "message": f"Invalid JSON: {line}"})
                continue

            # пока поддерживается только одна команда - "generate"
            if cmd.get("type") == "generate":
                # извлекаем параметры команды.
                # у каждого поля есть запасное значение на случай,
                # если приложение его не передало
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

                # если пользователь поменял число шагов или коэффициент
                # направления между генерациями - пересоздаём только
                # солвер и cfg. сами модели sdxl остаются в памяти -
                # перезагружать их каждый раз очень долго
                if steps != pipeline.num_steps:
                    pipeline.solver = EulerSolver(pipeline.sde, steps)
                    pipeline.num_steps = steps
                if guidance != pipeline.cfg.guidance_scale:
                    pipeline.cfg = ClassifierFreeGuidance(guidance)

                # создаём свежую временную папку под каждую генерацию.
                # туда сложим превью-миниатюры и финальную картинку
                tmp_dir = tempfile.mkdtemp(prefix="diffusion_gen_")
                # сообщаем приложению, что генерация началась и сколько
                # всего будет шагов - оно нарисует полосу прогресса
                send({"type": "generation_started", "total_steps": steps})

                # оборачиваем всю генерацию в try, чтобы ошибка
                # одной картинки не убивала весь процесс
                try:
                    start_time = time.time()

                    if seed is not None:
                        set_seed(seed)
                    generator = get_generator(seed, pipeline.device)

                    # повторяем тот же алгоритм, что в diffusion_pipeline.generate(),
                    # но с возможностью отправлять промежуточные превью
                    # после каждого шага - у обычного generate() такого хука нет

                    # кодируем текстовый запрос
                    cond_embeds, cond_pooled = pipeline.models.encode_prompt(prompt)
                    uncond_embeds, uncond_pooled = pipeline.models.encode_prompt(
                        negative_prompt
                    )
                    time_ids = pipeline._build_time_ids(height, width)

                    # создаём начальный шум в скрытом представлении
                    latent_shape = pipeline.model_config.get_latent_shape(height, width)
                    latents = randn_tensor(
                        latent_shape, pipeline.device, torch.float32, generator
                    )

                    # сбрасываем солвер и берём заранее посчитанную
                    # сетку моментов времени
                    pipeline._setup_solver(
                        cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
                    )
                    timesteps = pipeline.solver.timesteps

                    # выбираем три шага, после которых отправим превью-миниатюру:
                    # примерно треть, две трети и финал. это даёт
                    # пользователю ощущение, как картинка проявляется из шума
                    total = len(timesteps) - 1
                    preview_at = set()
                    if total >= 3:
                        preview_at = {total // 3, 2 * total // 3, total - 1}
                    elif total >= 1:
                        preview_at = {total - 1}

                    # флаг "мы на видеокарте apple". нужен для ручной
                    # очистки её памяти перед декодированием - vae
                    # выжирает много, и без очистки можно получить ошибку
                    is_mps = str(pipeline.device) == "mps"

                    # === цикл обратной диффузии ===
                    for i in range(total):
                        # текущий момент времени и следующий
                        t = timesteps[i].to(pipeline.device)
                        t_prev = timesteps[i + 1].to(pipeline.device)
                        # переводим непрерывный t в дискретный индекс для u-net
                        discrete_t = pipeline._continuous_to_discrete(t)

                        # предсказание шума через cfg (два прохода u-net,
                        # смешанные по правилу направления без классификатора)
                        noise_pred = pipeline._predict_noise(
                            latents,
                            discrete_t,
                            cond_embeds,
                            uncond_embeds,
                            cond_pooled,
                            uncond_pooled,
                            time_ids,
                        )

                        # один шаг метода Эйлера
                        latents = pipeline.solver.step(
                            latents.float(), t, t_prev, noise_pred.float()
                        )

                        # сообщение приложению о прогрессе
                        step_num = i + 1
                        msg = {
                            "type": "progress",
                            "step": step_num,
                            "total": total,
                        }

                        # если попали на контрольный шаг - декодируем латент
                        # в превью-миниатюру и кладём путь к файлу в сообщение
                        if step_num in preview_at:
                            # перед декодированием освобождаем память видеокарты
                            if is_mps:
                                torch.mps.empty_cache()
                            # vae превращает латент в полноразмерную картинку
                            img = pipeline._decode_and_postprocess(latents)
                            # для превью хватит маленького размера
                            thumb = img.resize((256, 256), Image.LANCZOS)
                            img_path = os.path.join(
                                tmp_dir, f"step_{step_num:04d}.jpg"
                            )
                            # jpeg для превью - вес минимальный
                            thumb.save(img_path, "JPEG", quality=75)
                            del img, thumb
                            if is_mps:
                                torch.mps.empty_cache()
                            msg["image"] = img_path

                        send(msg)

                    # финальное декодирование
                    if is_mps:
                        torch.mps.empty_cache()
                    final_img = pipeline._decode_and_postprocess(latents)
                    # финальную картинку сохраняем в png без потерь
                    final_path = os.path.join(tmp_dir, "final.png")
                    final_img.save(final_path)
                    del final_img
                    if is_mps:
                        torch.mps.empty_cache()

                    # сообщение об успехе: путь к файлу и время генерации
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
                    # ошибка одной генерации - сообщаем приложению,
                    # но не выходим из цикла. ждём следующую команду
                    send({"type": "error", "message": str(e)})
                    traceback.print_exc(file=sys.stderr)

    except Exception as e:
        # фатальная ошибка - что-то сломалось ещё до первой генерации.
        # сообщаем приложению и завершаем процесс с кодом 1
        send({"type": "error", "message": f"Fatal: {str(e)}"})
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
