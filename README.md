# Diffusion Pipeline

Курсовая работа по теме «Генерация изображения на основе диффузионных моделей нейросетей».

Предобученные компоненты SDXL (CLIP, VAE, U-Net) используются как строительные блоки, но весь диффузионный процесс — SDE, солвер, noise schedule — написан с нуля.

## Требования

- Python 3.10+
- Node.js 18+ и npm (для десктопного приложения)
- ~6.5 ГБ свободного места (модель SDXL скачается при первом запуске)
- macOS с Apple Silicon (MPS) / NVIDIA GPU (CUDA) / CPU

## Установка

```bash
# 1. Python-зависимости
pip install -r requirements.txt

# 2. Десктопное приложение (опционально)
cd app
npm install
```

## Запуск

### Десктопное приложение (Electron)

```bash
cd app
npm run dev
```

Откроется окно чата — после ввода промпта генерируется изображение с отображением промежуточных шагов.

### Из терминала

```bash
python3 generate.py "a majestic lion in the savannah, golden hour lighting"

# С параметрами
python3 generate.py "futuristic cityscape at night" --steps 20 --guidance 9.0 --seed 42
```

### Параметры CLI

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `prompt` | — | Текстовый промпт |
| `--steps` | 30 | Количество шагов |
| `--guidance` | 7.5 | CFG guidance scale |
| `--seed` | случайный | Seed для воспроизводимости |
| `--width` / `--height` | 1024 | Размер изображения |
| `--negative_prompt` | quality filter | Отрицательный промпт |
| `--output` | ./output | Директория для результатов |
| `--save_intermediates` | false | Сохранять промежуточные шаги |

## Что реализовано вручную

Предобученные модели (CLIP-L, OpenCLIP-G, VAE, U-Net) загружаются из HuggingFace. Всё остальное написано с нуля:

- **VP-SDE** — прямой и обратный стохастические процессы, маргинальное распределение, конвертация noise↔score
- **DPM-Solver++** — солвер 2-го порядка для probability flow ODE, 1 вызов нейросети на шаг
- **Scaled Linear Schedule** — расписание шума (линейное в пространстве √β)
- **Classifier-Free Guidance** — батчевый inference с условным и безусловным предсказаниями
- **Математическое ядро** — численная верификация уравнения Фоккера-Планка, SNR, log-SNR, score estimation
- **Пайплайн** — оркестрация всех компонентов, непрерывное время, конвертация в дискретные timesteps U-Net

## Тесты

```bash
python3 -m pytest tests/ -v
```

29 тестов: scheduler (граничные условия, монотонность), VP-SDE (drift, diffusion, маргинальное распределение), DPM-Solver++ (порядок, временная сетка, dynamic thresholding).

## Структура проекта

```
├── generate.py              # CLI-генерация
├── bridge.py                # Мост Python↔Electron
├── app/                     # Десктопное приложение (Electron + React + TypeScript)
│   ├── electron/            # Main process, preload
│   └── src/                 # React-компоненты, стили, типы
├── src/
│   ├── pipeline/            # Основной пайплайн генерации
│   ├── sde/                 # VP-SDE
│   ├── solvers/             # DPM-Solver++
│   ├── schedulers/          # Scaled Linear Schedule
│   ├── guidance/            # Classifier-Free Guidance
│   ├── models/              # Загрузка предобученных моделей
│   ├── math_core/           # Фоккер-Планк, SNR, score estimation
│   └── utils/               # Устройства, изображения, seed
├── tests/                   # pytest-тесты
└── requirements.txt
```

## Математика

**VP-SDE (прямой процесс):**

    dx = -0.5 β(t) x dt + √β(t) dw
    q(x_t | x_0) = N(√ᾱ(t) x_0, (1 - ᾱ(t)) I)

**DPM-Solver++** — экспоненциальный интегратор для reverse ODE в параметризации x₀. Мультишаговая экстраполяция 2-го порядка. Рекомендуемое количество шагов: 20–30.

**Scaled Linear Schedule** — β линейно в пространстве √β: β_min = 0.00085, β_max = 0.012.

## Платформа

Оптимизировано для macOS (Apple Silicon, MPS backend). Также работает на CUDA и CPU.

Примерная скорость на M2 Pro:
- 20 шагов: ~40–60 сек
- 30 шагов: ~60–90 сек
