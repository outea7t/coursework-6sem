"""
Classifier-Free Guidance (CFG).

CFG — техника для усиления влияния текстового условия на генерацию.
Вместо использования отдельного классификатора (как в classifier guidance),
CFG использует саму диффузионную модель, обученную с dropout условия.

Математика:
===========

    При обучении модель случайно (с вероятностью ~10%) получает пустой
    промпт вместо реального. Это позволяет модели предсказывать шум
    как с условием, так и без.

    При генерации:
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

    где:
        eps_uncond — предсказание шума без текстового условия
        eps_cond — предсказание шума с текстовым условием
        w — guidance scale (вес)

    Эквивалентная формулировка через score:
        score_guided = (1-w)*score_uncond + w*score_cond
                     = score_uncond + w*(score_cond - score_uncond)

    Это можно интерпретировать как движение в направлении
    более высокой вероятности текстового условия:
        score_guided ∝ nabla_x [log p(x) + w * log p(text|x)]

    Выбор guidance scale:
        w = 1.0: без guidance (обычная генерация)
        w = 7.5: стандартное значение для Stable Diffusion
        w = 10-15: агрессивное следование промпту
        w > 20: возможны артефакты (пересыщение цветов)

Оптимизация батчинга:
    Вместо двух отдельных forward pass (uncond + cond), мы объединяем
    их в один батч размера 2. Это в ~1.5x быстрее, чем два отдельных вызова.

Ссылки:
    Ho, J., & Salimans, T. (2022).
    "Classifier-Free Diffusion Guidance."
"""

import torch
from torch import Tensor


class ClassifierFreeGuidance:
    """Classifier-Free Guidance для SDXL.

    Выполняет guided sampling: усиливает влияние текстового условия
    на процесс генерации. Использует батчинг для эффективности.

    Args:
        guidance_scale: Вес guidance (w). По умолчанию 7.5.
    """

    def __init__(self, guidance_scale: float = 7.5) -> None:
        self.guidance_scale = guidance_scale

    def __call__(
        self,
        model: torch.nn.Module,
        x: Tensor,
        t: Tensor,
        cond_embeddings: Tensor,
        uncond_embeddings: Tensor,
        cond_pooled: Tensor,
        uncond_pooled: Tensor,
        time_ids: Tensor,
    ) -> Tensor:
        """Выполняет CFG-guided предсказание шума.

        Процесс:
        1. Объединяет unconditional и conditional входы в батч
        2. Один forward pass U-Net с batch_size=2
        3. Разделяет выход на uncond и cond
        4. Применяет CFG формулу

        Args:
            model: U-Net модель.
            x: Текущие латенты (1, 4, H/8, W/8).
            t: Текущий timestep.
            cond_embeddings: Условные текстовые эмбеддинги (1, 77, 2048).
            uncond_embeddings: Безусловные текстовые эмбеддинги (1, 77, 2048).
            cond_pooled: Pooled условные эмбеддинги (1, 1280).
            uncond_pooled: Pooled безусловные эмбеддинги (1, 1280).
            time_ids: SDXL time conditioning (1, 6).

        Returns:
            Guided предсказание шума (1, 4, H/8, W/8).
        """
        # Если guidance_scale == 1.0, CFG не нужен
        if self.guidance_scale == 1.0:
            added_cond_kwargs = {
                "text_embeds": cond_pooled,
                "time_ids": time_ids,
            }
            return model(
                x, t,
                encoder_hidden_states=cond_embeddings,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # Батчинг: объединяем uncond + cond в один forward pass
        latent_input = torch.cat([x, x])

        # Timestep: expand для батча
        if t.dim() == 0:
            t_input = t.unsqueeze(0).expand(2)
        else:
            t_input = torch.cat([t, t])

        # Конкатенация текстовых эмбеддингов: [uncond, cond]
        encoder_states = torch.cat([uncond_embeddings, cond_embeddings])

        # SDXL additional conditioning
        added_cond_kwargs = {
            "text_embeds": torch.cat([uncond_pooled, cond_pooled]),
            "time_ids": torch.cat([time_ids, time_ids]),
        }

        # Один forward pass с batch_size=2
        noise_pred = model(
            latent_input,
            t_input,
            encoder_hidden_states=encoder_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Разделение на unconditional и conditional предсказания
        noise_uncond, noise_cond = noise_pred.chunk(2)

        # CFG формула:
        # eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        # Эквивалентно: eps_guided = (1 - w) * eps_uncond + w * eps_cond
        noise_guided = noise_uncond + self.guidance_scale * (
            noise_cond - noise_uncond
        )

        return noise_guided
