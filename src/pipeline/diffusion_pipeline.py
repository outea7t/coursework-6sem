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
from ..solvers import EulerSolver
from ..schedulers import ScaledLinearScheduler
from ..guidance.cfg import ClassifierFreeGuidance
from ..utils.device import get_device, get_dtype, randn_tensor
from ..utils.seed import set_seed, get_generator
from ..utils.image_utils import tensor_to_pil, save_image

logger = logging.getLogger(__name__)


# создаем объекты расписания шума и солвера
def _create_scheduler():
    return ScaledLinearScheduler()

def _create_solver(sde, num_steps: int):
    return EulerSolver(sde, num_steps)


class DiffusionPipeline:

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "auto",
        dtype: str = "float16",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        # определяем устройство (видеокарта apple, nvidia или процессор)
        # и тип данных. float16 экономит память, но на некоторых операциях
        # приходится переключаться обратно в float32
        self.device = get_device(device)
        self.dtype = get_dtype(self.device) if dtype == "float16" else torch.float32
        self.num_steps = num_steps

        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        # загружаем все 4 компонента sdxl: два текстовых энкодера,
        # vae-декодер и unet. это самая долгая часть инициализации
        self.models = PretrainedModels(model_id, self.device, self.dtype)

        # собираем математическую часть. эти объекты связаны цепочкой:
        # scheduler описывает коэффициенты шума, sde - уравнение диффузии
        # на базе этих коэффициентов, solver - метод его численного решения
        self.scheduler = _create_scheduler()
        self.sde = VPSDE(scheduler=self.scheduler)
        self.solver = _create_solver(self.sde, num_steps)
        # cfg - обёртка над unet для вычисления направляющей подсказки
        self.cfg = ClassifierFreeGuidance(guidance_scale)
        # конфиг модели нужен для подсчёта размеров латентного пространства
        self.model_config = ModelConfig(model_id=model_id)

        logger.info(
            f"Pipeline initialized: Euler, "
            f"steps={num_steps}, guidance_scale={guidance_scale}"
        )

    # декоратор отключает подсчёт градиентов для всего метода -
    # мы только делаем вывод, обучать модель не собираемся
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
        # список промежуточных картинок - заполняется только если
        # пользователь попросил сохранять их через save_intermediates
        intermediates = []

        # set_seed фиксирует зерно в python, numpy и torch - это нужно,
        # чтобы одна и та же пара (промпт, seed) всегда давала одинаковую картинку.
        # generator - отдельный генератор для создания начального шума,
        # привязан к устройству
        if seed is not None:
            set_seed(seed)
        generator = get_generator(seed, self.device)

        # sdxl использует два текстовых энкодера одновременно (CLIP-L и OpenCLIP-G).
        # encode_prompt вызывает оба и склеивает результат.
        # на выходе две вещи: последовательность векторов по токенам промпта
        # и один сжатый вектор смысла целиком - оба нужны unet как условие
        logger.info("Encoding text prompt...")
        cond_embeds, cond_pooled = self.models.encode_prompt(prompt)
        # то же для негативного промпта - нужен для направляющей подсказки:
        # будем вычислять eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        uncond_embeds, uncond_pooled = self.models.encode_prompt(negative_prompt)

        # служебный тензор именно для sdxl: исходный размер картинки,
        # координаты обрезки и целевой размер - unet учитывает их как условие
        time_ids = self._build_time_ids(height, width)

        # создание начального шума
        # модели sdxl работают в сжатом латентном пространстве: для 1024x1024
        # картинки латент получается 128x128x4 (в 8 раз меньше по стороне,
        # 4 канала вместо 3)
        latent_shape = self.model_config.get_latent_shape(height, width)
        # чистый гауссов шум - отправная точка обратного процесса (x_T)
        latents = randn_tensor(latent_shape, self.device, torch.float32, generator)

        # сбрасываем внутреннее состояние солвера перед новой генерацией
        # на случай, если прошлый запуск что-то там оставил
        self._setup_solver(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        # основной цикл обратной диффузии 
        # timesteps - заранее рассчитанная сетка моментов времени:
        # от t=1 (чистый шум) до t≈0 (готовая картинка).
        # точек всегда на одну больше числа шагов (n+1 точек - n интервалов)
        timesteps = self.solver.timesteps
        logger.info(f"Starting reverse process: {self.num_steps} steps...")

        # показываем полоски прогресса
        # удобно при запуске из cli, в режиме клиент-сервера не используется
        for i in tqdm(range(len(timesteps) - 1), desc="Generating", unit="step"):
            # текущий момент времени и следующий (ближе к 0, так как идём назад)
            t = timesteps[i].to(self.device)
            t_prev = timesteps[i + 1].to(self.device)

            # unet обучен в дискретной сетке 0-999, а мы работаем в непрерывной
            # [0, 1] - переводим t в ближайший целый номер для unet
            discrete_t = self._continuous_to_discrete(t)

            # два прохода unet для направляющей подсказки 
            # внутри _predict_noise сеть вызывается дважды:
            # с условием (позитивный промпт) и без (негативный).
            # результат экстраполируется и определяет, куда двигать латент
            noise_pred = self._predict_noise(
                latents, discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )

            # один шаг эйлера 
            # приводим к float32 для численной стабильности:
            # в солвере деления на маленькие sigma, float16 даёт nan
            latents = self.solver.step(
                latents.float(), t, t_prev, noise_pred.float()
            )

            # если пользователь попросил промежуточные картинки - декодируем
            # латент в полноразмерное изображение каждые N шагов.
            # операция дорогая (vae тяжёлый), поэтому по интервалу, а не каждый шаг
            if save_intermediates and (i + 1) % intermediates_interval == 0:
                intermediate_img = self._decode_and_postprocess(latents)
                intermediates.append(intermediate_img)

        # финальное декодирование 
        # последний латент превращаем в полноразмерную картинку через vae
        logger.info("Decoding latents...")
        image = self._decode_and_postprocess(latents)

        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.1f}s")

        # возвращаем итоговую картинку и список промежуточных (может быть пуст)
        return image, intermediates

    # предсказание шума с направляющей подсказкой.
    # cfg внутри вызывает unet дважды (с условием и без) и комбинирует
    # результаты по формуле eps = eps_uncond + guidance * (eps_cond - eps_uncond).
    # приводим латент к нужному типу данных (float16 обычно), а результат
    # возвращаем в float32 для стабильности последующего шага солвера
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

    # фабрика замыкания model_fn(x, t) - упаковывает все параметры
    # генерации так, чтобы солвер мог дёргать модель, зная только (x, t).
    # нужно для адаптивных методов, которые сами вызывают модель внутри
    # своих шагов. у эйлера предсказание шума делается снаружи и
    # передаётся в solver.step() как аргумент, так что этот метод -
    # задел на случай подключения других солверов
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

    # подготовка солвера к новой генерации: передача model_fn (если солвер
    # умеет его принимать) и сброс внутреннего состояния (историю шагов
    # и т.п.). для эйлера оба блока либо пусты, либо no-op,
    # но оставлены универсальными для совместимости с другими солверами
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

    # служебный тензор для unet от sdxl. модель была обучена принимать
    # шестёрку чисел (высота_исх, ширина_исх, y_обрезки, x_обрезки, высота_цели,
    # ширина_цели) как дополнительное условие - чтобы уметь учитывать
    # соотношение сторон. мы не обрезаем и не меняем размер, поэтому
    # исходный и целевой совпадают, а координаты обрезки - нули
    def _build_time_ids(self, height: int, width: int) -> Tensor:
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=self.dtype,
            device=self.device,
        )
        return time_ids

    # перевод непрерывного времени t∈[0,1] в дискретный индекс 0-999.
    # unet обучался именно на целых номерах шагов, других не понимает.
    # round() а не int() - чтобы t=0.999 правильно превратилось в 999,
    # а не в 998. clamp - страховка от выхода за границу тензора из-за
    # численной погрешности
    def _continuous_to_discrete(self, t: Tensor) -> Tensor:
        num_timesteps = self.model_config.num_train_timesteps
        discrete = (t * (num_timesteps - 1)).round().long().clamp(0, num_timesteps - 1)
        return discrete.to(self.device)

    # декодирование латента в готовую картинку.
    # vae.decode превращает тензор 128x128x4 в 1024x1024x3, затем
    # tensor_to_pil нормализует значения в диапазон 0-255 и заворачивает
    # в объект PIL.Image, с которым уже можно делать resize/save
    def _decode_and_postprocess(self, latents: Tensor) -> Image.Image:
        image_tensor = self.models.decode_latents(latents.float())
        return tensor_to_pil(image_tensor)
