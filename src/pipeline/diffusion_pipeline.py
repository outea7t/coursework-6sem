# основной пайплайн генерации изображений

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
from ..solvers import EulerODESolver
from ..schedulers import ScaledLinearScheduler
from ..guidance.cfg import ClassifierFreeGuidance
from ..utils.device import get_device, get_dtype, randn_tensor
from ..utils.seed import set_seed, get_generator
from ..utils.image_utils import tensor_to_pil, save_image

logger = logging.getLogger(__name__)


def _create_scheduler():
    return ScaledLinearScheduler()


def _create_solver(sde, num_steps: int):
    return EulerODESolver(sde, num_steps)


class DiffusionPipeline:

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

        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        self.models = PretrainedModels(model_id, self.device, self.dtype)

        self.scheduler = _create_scheduler()
        self.sde = VPSDE(scheduler=self.scheduler)
        self.solver = _create_solver(self.sde, num_steps)
        self.cfg = ClassifierFreeGuidance(guidance_scale)
        self.model_config = ModelConfig(model_id=model_id)

        logger.info(
            f"Pipeline initialized: Euler ODE, "
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
        start_time = time.time()
        intermediates = []

        if seed is not None:
            set_seed(seed)
        generator = get_generator(seed, self.device)

        # кодирование текста
        logger.info("Encoding text prompt...")
        cond_embeds, cond_pooled = self.models.encode_prompt(prompt)
        uncond_embeds, uncond_pooled = self.models.encode_prompt(negative_prompt)

        time_ids = self._build_time_ids(height, width)

        # начальный шум
        latent_shape = self.model_config.get_latent_shape(height, width)
        latents = randn_tensor(latent_shape, self.device, torch.float32, generator)

        self._setup_solver(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        # обратный процесс
        timesteps = self.solver.timesteps
        logger.info(f"Starting reverse process: {self.num_steps} steps...")

        for i in tqdm(range(len(timesteps) - 1), desc="Generating", unit="step"):
            t = timesteps[i].to(self.device)
            t_prev = timesteps[i + 1].to(self.device)

            discrete_t = self._continuous_to_discrete(t)

            noise_pred = self._predict_noise(
                latents, discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )

            latents = self.solver.step(
                latents.float(), t, t_prev, noise_pred.float()
            )

            if save_intermediates and (i + 1) % intermediates_interval == 0:
                intermediate_img = self._decode_and_postprocess(latents)
                intermediates.append(intermediate_img)

        # декодирование
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
        model_fn = self._make_model_fn(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        if hasattr(self.solver, 'set_model_fn'):
            self.solver.set_model_fn(model_fn)

        if hasattr(self.solver, 'reset'):
            self.solver.reset()

    def _build_time_ids(self, height: int, width: int) -> Tensor:
        # sdxl time_ids: original_size, crop, target_size
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=self.dtype,
            device=self.device,
        )
        return time_ids

    def _continuous_to_discrete(self, t: Tensor) -> Tensor:
        # t in [eps, 1] -> дискретный timestep 0-999
        num_timesteps = self.model_config.num_train_timesteps
        discrete = (t * (num_timesteps - 1)).round().long().clamp(0, num_timesteps - 1)
        return discrete.to(self.device)

    def _decode_and_postprocess(self, latents: Tensor) -> Image.Image:
        image_tensor = self.models.decode_latents(latents.float())
        return tensor_to_pil(image_tensor)
