"""
Основной диффузионный пайплайн для генерации изображений.

Соединяет все компоненты: предобученные модели (CLIP, VAE, U-Net),
SDE, солверы, schedulers и Classifier-Free Guidance в единый
пайплайн генерации text-to-image.

Процесс генерации:
===================

    1. Кодирование текста:
       prompt → [CLIP-L, OpenCLIP-G] → (prompt_embeds, pooled_embeds)

    2. Инициализация из шума:
       x_T ~ N(0, I) в латентном пространстве (1, 4, 128, 128)

    3. Обратный процесс (решение reverse SDE/ODE):
       for t = T → 0:
           noise_pred = CFG(U-Net(x_t, t, text_embeds))
           x_{t-dt} = solver.step(x_t, t, t_prev, noise_pred)

    4. Декодирование:
       x_0 → VAE decoder → изображение (1, 3, 1024, 1024)

    5. Постобработка:
       [-1, 1] → [0, 255] → PIL Image → сохранение
"""

import logging
import time
from typing import Callable

import torch
from torch import Tensor
from PIL import Image
from tqdm import tqdm

from ..models.pretrained_loader import PretrainedModels
from ..models.model_config import ModelConfig
from ..sde import VPSDE
from ..solvers import DPMSolverPP
from ..schedulers import ScaledLinearScheduler
from ..guidance.cfg import ClassifierFreeGuidance
from ..utils.device import get_device, get_dtype, randn_tensor
from ..utils.seed import set_seed, get_generator
from ..utils.image_utils import tensor_to_pil, save_image

logger = logging.getLogger(__name__)


def _create_scheduler():
    """Создаёт noise scheduler (Scaled Linear, как в Stable Diffusion)."""
    return ScaledLinearScheduler()


def _create_solver(sde, num_steps: int):
    """Создаёт DPM-Solver++ солвер."""
    return DPMSolverPP(sde, num_steps)


class DiffusionPipeline:
    """Основной пайплайн text-to-image генерации.

    Args:
        model_id: HuggingFace model ID.
        device: Устройство ("auto", "mps", "cuda", "cpu").
        dtype: Тип данных ("float16", "float32").
        scheduler_name: Имя noise scheduler.
        solver_name: Имя солвера.
        num_steps: Количество шагов.
        guidance_scale: Вес Classifier-Free Guidance.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "auto",
        dtype: str = "float16",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        self.device = get_device(device)
        self.dtype = get_dtype(self.device) if dtype == "float16" else torch.float32
        self.num_steps = num_steps

        # Загрузка предобученных моделей
        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        self.models = PretrainedModels(model_id, self.device, self.dtype)

        # Noise scheduler (Scaled Linear — Stable Diffusion)
        self.scheduler = _create_scheduler()

        # VP-SDE — основное для Stable Diffusion
        self.sde = VPSDE(scheduler=self.scheduler)

        # DPM-Solver++ — 2-го порядка, 1 NFE/шаг
        self.solver = _create_solver(self.sde, num_steps)

        # Classifier-Free Guidance
        self.cfg = ClassifierFreeGuidance(guidance_scale)

        # Model config
        self.model_config = ModelConfig(model_id=model_id)

        logger.info(
            f"Pipeline initialized: DPM-Solver++, "
            f"steps={num_steps}, guidance_scale={guidance_scale}"
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
        save_intermediates: bool = False,
        intermediates_interval: int = 5,
    ) -> tuple[Image.Image, list[Image.Image]]:
        """Генерация изображения по текстовому промпту.

        Args:
            prompt: Текстовый промпт.
            negative_prompt: Отрицательный промпт.
            seed: Значение seed для воспроизводимости.
            height: Высота изображения.
            width: Ширина изображения.
            save_intermediates: Сохранять промежуточные шаги.
            intermediates_interval: Интервал сохранения промежуточных шагов.

        Returns:
            (image, intermediates): Итоговое изображение и список промежуточных.
        """
        start_time = time.time()
        intermediates = []

        # 1. Установка seed
        if seed is not None:
            set_seed(seed)
        generator = get_generator(seed, self.device)

        # 2. Кодирование текста (оба энкодера SDXL)
        logger.info("Encoding text prompt...")
        cond_embeds, cond_pooled = self.models.encode_prompt(prompt)
        uncond_embeds, uncond_pooled = self.models.encode_prompt(negative_prompt)

        # 3. Подготовка SDXL time_ids (original_size, crop, target_size)
        time_ids = self._build_time_ids(height, width)

        # 4. Начальный шум в латентном пространстве
        latent_shape = self.model_config.get_latent_shape(height, width)
        latents = randn_tensor(latent_shape, self.device, torch.float32, generator)

        # Масштабирование начального шума (для schedulers с нестандартным prior)
        # Для VP-SDE prior = N(0, I), масштабирование не нужно

        # 5. Подготовка солвера
        self._setup_solver(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        # 6. Обратный процесс — интегрирование reverse SDE/ODE
        timesteps = self.solver.timesteps
        logger.info(f"Starting reverse process: {self.num_steps} steps...")

        for i in tqdm(range(len(timesteps) - 1), desc="Generating", unit="step"):
            t = timesteps[i].to(self.device)
            t_prev = timesteps[i + 1].to(self.device)

            # Конвертация непрерывного t -> дискретный timestep для U-Net
            discrete_t = self._continuous_to_discrete(t)

            # Предсказание шума через CFG
            noise_pred = self._predict_noise(
                latents, discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )

            # Шаг солвера (в float32 для численной стабильности)
            latents = self.solver.step(
                latents.float(), t, t_prev, noise_pred.float()
            )

            # Сохранение промежуточных шагов
            if save_intermediates and (i + 1) % intermediates_interval == 0:
                intermediate_img = self._decode_and_postprocess(latents)
                intermediates.append(intermediate_img)

        # 7. Декодирование латентов в изображение
        logger.info("Decoding latents...")
        image = self._decode_and_postprocess(latents)

        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.1f}s")

        return image, intermediates

    def _predict_noise(
        self,
        latents: Tensor,
        timestep: Tensor,
        cond_embeds: Tensor,
        uncond_embeds: Tensor,
        cond_pooled: Tensor,
        uncond_pooled: Tensor,
        time_ids: Tensor,
    ) -> Tensor:
        """Предсказание шума через U-Net + CFG.

        Конвертирует латенты в float16 для U-Net, затем обратно в float32.

        Args:
            latents: Текущие латенты (float32).
            timestep: Дискретный timestep для U-Net.
            Остальные аргументы — текстовые эмбеддинги и условия SDXL.

        Returns:
            Предсказание шума (float32).
        """
        # U-Net работает в float16
        noise_pred = self.cfg(
            self.models.unet,
            latents.to(self.dtype),
            timestep,
            cond_embeds,
            uncond_embeds,
            cond_pooled,
            uncond_pooled,
            time_ids,
        )

        return noise_pred.float()

    def _make_model_fn(
        self,
        cond_embeds: Tensor,
        uncond_embeds: Tensor,
        cond_pooled: Tensor,
        uncond_pooled: Tensor,
        time_ids: Tensor,
    ) -> Callable:
        """Создаёт callable для солверов, требующих промежуточных оценок.

        Возвращает функцию model_fn(x, t) -> noise_prediction,
        которая включает CFG и конвертацию timesteps.

        Args:
            Текстовые эмбеддинги и условия SDXL.

        Returns:
            Callable(x, t) -> noise_pred.
        """
        def model_fn(x: Tensor, t: Tensor) -> Tensor:
            discrete_t = self._continuous_to_discrete(t)
            return self._predict_noise(
                x, discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )
        return model_fn

    def _setup_solver(
        self,
        cond_embeds: Tensor,
        uncond_embeds: Tensor,
        cond_pooled: Tensor,
        uncond_pooled: Tensor,
        time_ids: Tensor,
    ) -> None:
        """Настраивает солвер перед генерацией.

        Для солверов, требующих model_fn (Heun, RK4, Adaptive),
        устанавливает callable функцию модели.
        Для DPM-Solver++ сбрасывает историю.
        """
        model_fn = self._make_model_fn(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        if hasattr(self.solver, 'set_model_fn'):
            self.solver.set_model_fn(model_fn)

        if hasattr(self.solver, 'reset'):
            self.solver.reset()

    def _build_time_ids(self, height: int, width: int) -> Tensor:
        """Создаёт SDXL time_ids для кондиционирования размера.

        SDXL U-Net принимает дополнительную информацию о:
        - Оригинальном размере изображения (original_height, original_width)
        - Координатах обрезки (crop_top, crop_left)
        - Целевом размере (target_height, target_width)

        Args:
            height: Высота целевого изображения.
            width: Ширина целевого изображения.

        Returns:
            time_ids тензор (1, 6).
        """
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=self.dtype,
            device=self.device,
        )
        return time_ids

    def _continuous_to_discrete(self, t: Tensor) -> Tensor:
        """Конвертация непрерывного времени t в дискретный timestep для U-Net.

        U-Net Stable Diffusion обучен на дискретных timesteps 0-999.
        Наши SDE работают с непрерывным t in [epsilon, 1].

        Маппинг: discrete = round(t * (num_train_timesteps - 1))

        Args:
            t: Непрерывное время, скаляр.

        Returns:
            Дискретный timestep для U-Net.
        """
        num_timesteps = self.model_config.num_train_timesteps
        discrete = (t * (num_timesteps - 1)).round().long().clamp(0, num_timesteps - 1)
        return discrete.to(self.device)

    def _decode_and_postprocess(self, latents: Tensor) -> Image.Image:
        """Декодирует латенты и конвертирует в PIL Image.

        Args:
            latents: Латентный тензор (1, 4, H/8, W/8).

        Returns:
            PIL Image.
        """
        image_tensor = self.models.decode_latents(latents.float())
        return tensor_to_pil(image_tensor)
