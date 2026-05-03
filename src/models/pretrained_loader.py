# загрузка предобученных компонентов модели sdxl
# здесь мы НЕ обучаем сети - только подгружаем готовые веса
# и используем их как чёрные ящики

import logging
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class PretrainedModels:
    # хранилище для трёх компонентов sdxl:
    # 1) текстовые кодировщики (CLIP-L и OpenCLIP-G) - превращают
    #    текстовый запрос в вектор чисел;
    # 2) автокодировщик изображений (VAE) - переводит картинку
    #    из обычного пространства пикселей в скрытое и обратно;
    # 3) сеть предсказания шума (U-Net) - на каждом шаге диффузии
    #    смотрит на зашумлённое скрытое представление и угадывает,
    #    какой в нём шум

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        logger.info(f"Loading SDXL components from {model_id}...")
        # подгружаем компоненты по очереди - так удобнее показывать
        # прогресс в окне загрузки
        self._load_text_encoders(model_id)
        self._load_vae(model_id)
        self._load_unet(model_id)
        logger.info("All SDXL components loaded successfully.")

    def _load_text_encoders(self, model_id: str) -> None:
        # sdxl использует сразу два текстовых кодировщика:
        # CLIP-L (поменьше) и OpenCLIP-G (побольше).
        # их выходы потом склеиваются - получается богаче условие
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        logger.info("Loading CLIP-L text encoder...")
        # tokenizer разбивает текст на токены (примерно слова)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        # text_encoder превращает токены в вектор чисел.
        # .eval() переводит сеть в режим вывода (без обучения)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=self.dtype
        ).to(self.device).eval()

        logger.info("Loading OpenCLIP-G text encoder...")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=self.dtype
        ).to(self.device).eval()

    def _load_vae(self, model_id: str) -> None:
        from diffusers import AutoencoderKL

        # vae обязательно держим в float32. в float16 у него возникают
        # переполнения - получается nan и картинка чёрная
        logger.info("Loading VAE (float32)...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)

    def _load_unet(self, model_id: str) -> None:
        from diffusers import UNet2DConditionModel

        # u-net - самая большая сеть, именно она предсказывает шум.
        # это самая медленная операция всего цикла диффузии
        logger.info("Loading U-Net...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=self.dtype
        ).to(self.device).eval()

    @torch.no_grad()
    def encode_prompt(
        self, prompt: str
    ) -> Tuple[Tensor, Tensor]:
        # перевод текстового запроса в числа.
        # на вход обычная строка, на выходе два тензора:
        # 1) последовательность векторов по токенам (длина 77, размер 2048)
        # 2) один обобщённый вектор всего смысла (размер 1280)

        # CLIP-L: разбиваем текст на токены и пропускаем через сеть
        tokens_1 = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        output_1 = self.text_encoder(
            tokens_1.input_ids.to(self.device)
        )
        hidden_states_1 = output_1.last_hidden_state

        # OpenCLIP-G: то же самое, но эта сеть заодно отдаёт
        # обобщённый вектор. output_hidden_states=True нужен,
        # чтобы получить промежуточный слой - именно его
        # (а не финальный) использует sdxl
        tokens_2 = self.tokenizer_2(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        output_2 = self.text_encoder_2(
            tokens_2.input_ids.to(self.device),
            output_hidden_states=True,
        )
        hidden_states_2 = output_2.hidden_states[-2]
        pooled_output = output_2.text_embeds

        # склеиваем выходы двух кодировщиков по последней оси:
        # (1, 77, 768) + (1, 77, 1280) -> (1, 77, 2048)
        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)

        return prompt_embeds, pooled_output

    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        # vae-декодер: из скрытого представления (128x128x4)
        # получаем полноразмерную картинку (1024x1024x3).
        # деление на scaling_factor - нормировка, заложенная при обучении vae
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.float()).sample
        return image

    @torch.no_grad()
    def predict_noise(
        self,
        latents: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        added_cond_kwargs: dict,
    ) -> Tensor:
        # вызов u-net: на вход идёт зашумлённое скрытое представление,
        # текущий шаг времени и условия (текстовые векторы).
        # на выходе - предсказанный шум той же формы, что и латент
        noise_pred = self.unet(
            latents.to(self.dtype),
            timestep,
            encoder_hidden_states=encoder_hidden_states.to(self.dtype),
            added_cond_kwargs={
                k: v.to(self.dtype) for k, v in added_cond_kwargs.items()
            },
        ).sample

        return noise_pred
