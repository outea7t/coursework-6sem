# Diffusion Pipeline — Text-to-Image Generation

Авторская реализация text-to-image диффузионного пайплайна на основе стохастических и обыкновенных дифференциальных уравнений.

Курсовая работа по теме "Модели нейронных сетей" (подтема: диффузионные модели).

## Архитектура

Проект использует предобученные компоненты SDXL (CLIP, VAE, U-Net) как строительные блоки, но весь диффузионный процесс — SDE, солверы, schedulers — реализован самостоятельно.

### Компоненты

- **SDE** (`src/sde/`): VP-SDE, VE-SDE, Sub-VP SDE — стохастические дифференциальные уравнения прямого и обратного процесса
- **Солверы** (`src/solvers/`): Euler-Maruyama, Euler ODE (DDIM), Heun, RK4, DPM-Solver++, Adaptive RK45
- **Schedulers** (`src/schedulers/`): Linear, Cosine, Scaled Linear, Continuous
- **CFG** (`src/guidance/`): Classifier-Free Guidance
- **Math Core** (`src/math_core/`): Теоретические функции, верификация Фоккера-Планка

## Установка

```bash
pip install -r requirements.txt
```

При первом запуске модель SDXL (~6.5 ГБ) будет загружена из HuggingFace.

## Использование

```bash
# Базовая генерация
python3 generate.py "a majestic lion in the savannah, golden hour lighting"

# С выбором солвера и количества шагов
python3 generate.py "futuristic cityscape at night" --solver runge_kutta --steps 50

# Быстрая генерация через DPM-Solver++
python3 generate.py "portrait of a wizard, fantasy art" --solver dpm_solver_pp --steps 20

# С высоким guidance scale
python3 generate.py "a white cat wearing sunglasses" --guidance 12.0

# Воспроизводимый результат
python3 generate.py "a red rose with morning dew" --solver euler_ode --seed 42

# Адаптивный солвер
python3 generate.py "mountain landscape, photorealistic" --solver adaptive
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `prompt` | — | Текстовый промпт (обязательный) |
| `--negative_prompt` | quality filter | Отрицательный промпт |
| `--steps` | 30 | Количество шагов |
| `--solver` | dpm_solver_pp | Солвер: euler_maruyama, euler_ode, heun, runge_kutta, dpm_solver_pp, adaptive |
| `--scheduler` | scaled_linear | Расписание шума: linear, cosine, scaled_linear, continuous |
| `--guidance` | 7.5 | CFG guidance scale |
| `--seed` | random | Seed для воспроизводимости |
| `--width` | 1024 | Ширина изображения |
| `--height` | 1024 | Высота изображения |
| `--output` | ./output | Директория для результатов |
| `--save_intermediates` | false | Сохранять промежуточные шаги |

## Солверы

| Солвер | Тип | Порядок | NFE/шаг | Рекомендуемые шаги |
|--------|-----|---------|---------|-------------------|
| Euler-Maruyama | SDE (стохастический) | 1 | 1 | 50-100 |
| Euler ODE | ODE (детерминированный) | 1 | 1 | 50-100 |
| Heun | ODE (предиктор-корректор) | 2 | 2 | 30-50 |
| Runge-Kutta 4 | ODE | 4 | 4 | 20-30 |
| DPM-Solver++ | ODE (специализированный) | 2 | 1 | 15-25 |
| Adaptive RK45 | ODE (адаптивный) | 5 | 6-7 | автоматически |

## Тесты

```bash
pytest tests/ -v
```

## Платформа

Оптимизирован для macOS (Apple M2 Pro, MPS backend). Также работает на CUDA и CPU.

Ожидаемая производительность на M2 Pro:
- SDXL 1024×1024, 20 шагов DPM-Solver++: ~40–60 секунд
- SDXL 1024×1024, 30 шагов DPM-Solver++: ~60–90 секунд
