# Diffusion Pipeline — Text-to-Image Generation

Авторская реализация text-to-image диффузионного пайплайна на основе стохастических и обыкновенных дифференциальных уравнений.

Курсовая работа по теме "Модели нейронных сетей" (подтема: диффузионные модели).

## Архитектура

Проект использует предобученные компоненты SDXL (CLIP, VAE, U-Net) как строительные блоки, но весь диффузионный процесс — SDE, солвер, scheduler — реализован самостоятельно.

### Компоненты

| Модуль | Описание |
|--------|----------|
| `src/sde/vp_sde.py` | VP-SDE — стохастическое дифференциальное уравнение прямого и обратного процесса |
| `src/solvers/dpm_solver.py` | DPM-Solver++ — солвер 2-го порядка для диффузионных ODE (1 NFE/шаг) |
| `src/schedulers/scaled_linear_scheduler.py` | Scaled Linear noise schedule (как в Stable Diffusion) |
| `src/guidance/cfg.py` | Classifier-Free Guidance |
| `src/pipeline/diffusion_pipeline.py` | Основной пайплайн, соединяющий все компоненты |
| `src/models/` | Загрузка предобученных моделей SDXL (CLIP-L + OpenCLIP-G, VAE, U-Net) |
| `src/math_core/` | Теоретические функции: верификация Фоккера-Планка, SNR, score estimation |
| `src/utils/` | Утилиты: устройства (MPS/CUDA/CPU), сохранение изображений, seed |

### Процесс генерации

1. **Кодирование текста**: prompt → CLIP-L + OpenCLIP-G → эмбеддинги (2048-dim)
2. **Инициализация**: x_T ~ N(0, I) в латентном пространстве (1, 4, 128, 128)
3. **Обратный процесс**: DPM-Solver++ интегрирует reverse ODE от T к 0
4. **Декодирование**: VAE decoder → изображение (1024×1024)

## Установка

```bash
pip install -r requirements.txt
```

При первом запуске модель SDXL (~6.5 ГБ) будет загружена из HuggingFace.

## Использование

```bash
# Базовая генерация
python3 generate.py "a majestic lion in the savannah, golden hour lighting"

# С заданным количеством шагов
python3 generate.py "futuristic cityscape at night" --steps 50

# Быстрая генерация (20 шагов)
python3 generate.py "portrait of a wizard, fantasy art" --steps 20

# С высоким guidance scale
python3 generate.py "a white cat wearing sunglasses" --guidance 12.0

# Воспроизводимый результат
python3 generate.py "a red rose with morning dew" --seed 42

# Сохранение промежуточных шагов
python3 generate.py "mountain landscape" --save_intermediates --intermediates_interval 5
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `prompt` | — | Текстовый промпт (обязательный) |
| `--negative_prompt` | quality filter | Отрицательный промпт |
| `--steps` | 30 | Количество шагов DPM-Solver++ |
| `--guidance` | 7.5 | CFG guidance scale |
| `--seed` | random | Seed для воспроизводимости |
| `--width` | 1024 | Ширина изображения |
| `--height` | 1024 | Высота изображения |
| `--output` | ./output | Директория для результатов |
| `--save_intermediates` | false | Сохранять промежуточные шаги |
| `--intermediates_interval` | 5 | Интервал сохранения промежуточных шагов |
| `--model` | stabilityai/sdxl-base-1.0 | HuggingFace model ID |
| `--config` | config/default.yaml | Путь к конфигурации |
| `--verbose` | false | Подробное логирование |

## Математическая основа

### VP-SDE (Variance Preserving SDE)

Прямой процесс:

    dx = -0.5 * β(t) * x * dt + √β(t) * dw

Маргинальное распределение: q(x_t | x_0) = N(sqrt(ᾱ(t)) * x_0, (1 - ᾱ(t)) * I)

### DPM-Solver++ (Lu et al., 2022)

Экспоненциальный интегратор для probability flow ODE. Работает в пространстве data prediction (x₀), мультишаговая экстраполяция для 2-го порядка при 1 вызове U-Net на шаг.

Рекомендуемое количество шагов: 20–30.

### Scaled Linear Schedule

Линейная интерполяция в пространстве √β: β_min = 0.00085, β_max = 0.012 (параметры Stable Diffusion).

## Тесты

```bash
python3 -m pytest tests/ -v
```

29 тестов: свойства scheduler (граничные условия, монотонность, дискретные значения), VP-SDE (drift, diffusion, маргинальное распределение, обратный процесс), DPM-Solver++ (порядок, временная сетка, dynamic thresholding, шаг солвера).

## Структура проекта

```
├── generate.py                  # CLI для генерации изображений
├── config/default.yaml          # Конфигурация по умолчанию
├── src/
│   ├── pipeline/
│   │   └── diffusion_pipeline.py
│   ├── sde/
│   │   └── vp_sde.py
│   ├── solvers/
│   │   └── dpm_solver.py
│   ├── schedulers/
│   │   └── scaled_linear_scheduler.py
│   ├── guidance/
│   │   └── cfg.py
│   ├── models/
│   │   ├── model_config.py
│   │   └── pretrained_loader.py
│   ├── math_core/
│   │   ├── fokker_planck.py
│   │   ├── sde_theory.py
│   │   └── score_estimation.py
│   └── utils/
│       ├── device.py
│       ├── image_utils.py
│       └── seed.py
├── tests/
│   ├── test_schedulers.py
│   ├── test_sde.py
│   └── test_solvers.py
└── requirements.txt
```

## Платформа

Оптимизирован для macOS (Apple M2 Pro, MPS backend). Также работает на CUDA и CPU.

Ожидаемая производительность на M2 Pro:
- SDXL 1024×1024, 20 шагов: ~40–60 секунд
- SDXL 1024×1024, 30 шагов: ~60–90 секунд

## Зависимости

- PyTorch >= 2.0
- Transformers >= 4.30 (текстовые энкодеры CLIP)
- Diffusers >= 0.25 (загрузка предобученных моделей)
- Accelerate >= 0.20
- Pillow, PyYAML, tqdm, numpy
