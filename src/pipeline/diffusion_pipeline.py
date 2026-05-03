# главный класс - тот, в котором собирается вся генерация целиком.
# по шагам:
#   1) кодируем текст запроса в числа;
#   2) делаем стартовый шум;
#   3) крутим N шагов метода Эйлера (на каждом шаге сеть угадывает шум,
#      а мы немного приближаем картинку к чистой);
#   4) расшифровываем результат через vae - получается обычная картинка;
#   5) отдаём её наружу.

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


# вынес создание расписания и солвера в отдельные функции -
# если потом захочется заменить, например, на другой метод,
# не придётся лезть в большой класс
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
        # определяем, на чём будем считать (видеокарта apple, nvidia или процессор)
        # и каким типом чисел. float16 экономит память и быстрее на видеокарте,
        # но в чувствительных местах (vae, шаг солвера) переходим в float32 -
        # иначе появляется nan
        self.device = get_device(device)
        if dtype == "float16":
            self.dtype = get_dtype(self.device)
        else:
            self.dtype = torch.float32
        self.num_steps = num_steps

        logger.info(f"Device: {self.device}, dtype: {self.dtype}")
        # подгружаем все 4 предобученных компонента sdxl: два текстовых
        # кодировщика (CLIP-L и OpenCLIP-G), vae-декодер и u-net.
        # самая долгая часть инициализации - первый запуск ещё и
        # скачивает веса из интернета (~6.5 ГБ)
        self.models = PretrainedModels(model_id, self.device, self.dtype)

        # три объекта, на которых держится математика.
        # расписание знает, сколько шума должно быть на каждом моменте времени.
        # sde - это само уравнение зашумления.
        # solver - метод Эйлера, которым мы это уравнение решаем в обратную сторону
        self.scheduler = _create_scheduler()
        self.sde = VPSDE(scheduler=self.scheduler)
        self.solver = _create_solver(self.sde, num_steps)
        # cfg - то, благодаря чему картинка слушается запроса.
        # внутри умеет вызывать сеть с двумя текстами разом
        self.cfg = ClassifierFreeGuidance(guidance_scale)
        # тут лежат базовые параметры модели - нужны, чтобы посчитать,
        # какого размера должен быть массив шума для нужной картинки
        self.model_config = ModelConfig(model_id=model_id)

        logger.info(
            f"Pipeline initialized: Euler, "
            f"steps={num_steps}, guidance_scale={guidance_scale}"
        )

    # декоратор отключает подсчёт градиентов на весь метод -
    # мы только генерируем, обучать сеть не собираемся, а без
    # этого декоратора torch тратит время и память на лишний граф вычислений
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
        # сюда будут складываться промежуточные картинки -
        # только если пользователь попросил их сохранять
        intermediates = []

        # set_seed выставляет зерно во все библиотеки случайных чисел -
        # одна и та же пара (запрос, seed) всегда даёт одинаковую картинку.
        # generator - отдельный генератор для начального шума
        if seed is not None:
            set_seed(seed)
        generator = get_generator(seed, self.device)

        # переводим текст запроса в числа.
        # encode_prompt прогоняет его через два кодировщика sdxl и склеивает результат.
        # получаем две вещи: вектор по каждому токену и один общий вектор смысла -
        # оба пригодятся u-net как подсказка
        logger.info("Encoding text prompt...")
        cond_embeds, cond_pooled = self.models.encode_prompt(prompt)
        # то же самое для негативного запроса (что не должно быть на картинке).
        # понадобится в направлении без классификатора
        uncond_embeds, uncond_pooled = self.models.encode_prompt(negative_prompt)

        # маленький служебный тензор, который ждёт u-net в sdxl:
        # исходный размер, обрезка (у нас нулевая) и целевой размер.
        # модель училась учитывать это как доп. условие
        time_ids = self._build_time_ids(height, width)

        # делаем стартовый шум.
        # sdxl работает не с пикселями, а со скрытым представлением -
        # массивом, в 8 раз меньше картинки по стороне и с 4 каналами вместо 3.
        # для картинки 1024x1024 это будет 128x128x4
        latent_shape = self.model_config.get_latent_shape(height, width)
        # обычный гауссов шум - именно с него и начинается обратная диффузия
        latents = randn_tensor(latent_shape, self.device, torch.float32, generator)

        # сбрасываем солвер на случай, если прошлый запуск что-то в нём оставил
        self._setup_solver(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        # сам цикл обратной диффузии.
        # timesteps - готовая сетка моментов времени от t=1 (шум)
        # до t≈0 (картинка). точек на одну больше числа шагов:
        # для 30 шагов - 31 точка
        timesteps = self.solver.timesteps
        logger.info(f"Starting reverse process: {self.num_steps} steps...")

        # tqdm рисует полосу прогресса в терминале, на саму генерацию не влияет
        for i in tqdm(range(len(timesteps) - 1), desc="Generating", unit="step"):
            # текущий момент и следующий (он ближе к нулю - идём назад во времени)
            t = timesteps[i].to(self.device)
            t_prev = timesteps[i + 1].to(self.device)

            # u-net училась на целых номерах шагов от 0 до 999,
            # а у нас t лежит в [0, 1] - переводим к ближайшему целому
            discrete_t = self._continuous_to_discrete(t)

            # сеть угадывает, какой шум сидит в текущем латенте.
            # внутри _predict_noise сеть вызывается два раза - с описанием
            # и без - и эти ответы смешиваются (направление без классификатора)
            noise_pred = self._predict_noise(
                latents, discrete_t,
                cond_embeds, uncond_embeds,
                cond_pooled, uncond_pooled,
                time_ids,
            )

            # один шаг метода Эйлера - чуть приближаемся к чистой картинке.
            # переводим в float32, потому что в float16 маленькие коэффициенты
            # округляются до нуля и всё ломается
            latents = self.solver.step(
                latents.float(), t, t_prev, noise_pred.float()
            )

            # если попросили сохранять промежуточные кадры - расшифровываем
            # текущий латент в картинку. vae тяжёлый, поэтому делаем
            # не каждый шаг, а раз в несколько
            if save_intermediates and (i + 1) % intermediates_interval == 0:
                intermediate_img = self._decode_and_postprocess(latents)
                intermediates.append(intermediate_img)

        # последний латент расшифровываем через vae - получается готовая картинка
        logger.info("Decoding latents...")
        image = self._decode_and_postprocess(latents)

        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.1f}s")

        # сохранение делается уже снаружи, в generate.py
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
        # маленькая обёртка - просит cfg вызвать u-net с двумя текстами
        # и смешать ответы. на вход cfg даём латент в нужном типе чисел
        # (обычно float16), а наружу отдаём float32 - так шаг солвера
        # ведёт себя устойчивее
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
        # просто заворачиваем все аргументы в маленькую функцию (x, t) -> шум.
        # самому Эйлеру это не нужно (он зовёт сеть один раз в основном цикле),
        # но если потом подключить другой метод, который сам вызывает сеть
        # внутри шага - ему достаточно отдать вот эту функцию
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
        # перед каждой новой генерацией готовим солвер: даём ему
        # функцию для вызова сети и сбрасываем внутреннее состояние.
        # у Эйлера и то, и другое - пустышка, но если когда-то поменяю
        # солвер на более сложный, ничего переписывать не придётся
        model_fn = self._make_model_fn(
            cond_embeds, uncond_embeds, cond_pooled, uncond_pooled, time_ids
        )

        if hasattr(self.solver, 'set_model_fn'):
            self.solver.set_model_fn(model_fn)

        if hasattr(self.solver, 'reset'):
            self.solver.reset()

    def _build_time_ids(self, height: int, width: int) -> Tensor:
        # u-net от sdxl на вход кроме всего прочего ждёт ещё шестёрку чисел:
        # исходный размер, координаты обрезки и целевой размер.
        # это её особенность - так модель училась учитывать соотношение сторон.
        # мы ничего не обрезаем и не меняем размер - поэтому исходный = целевой,
        # а координаты обрезки нулевые
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=self.dtype,
            device=self.device,
        )
        return time_ids

    def _continuous_to_discrete(self, t: Tensor) -> Tensor:
        # перевод непрерывного времени t из [0, 1] в дискретный
        # индекс из 0..999. u-net учился именно на целых номерах шагов.
        # round() (а не int() или long()) важен: t=0.999 должно
        # стать 999, а не 998 (отбрасывание дробной части дало бы 998).
        # clamp - страховка от выхода за границы из-за погрешностей с плавающей точкой
        num_timesteps = self.model_config.num_train_timesteps
        discrete = (t * (num_timesteps - 1)).round().long().clamp(0, num_timesteps - 1)
        return discrete.to(self.device)

    def _decode_and_postprocess(self, latents: Tensor) -> Image.Image:
        # vae берёт скрытое представление и достаёт из него картинку.
        # tensor_to_pil потом приводит числа к диапазону 0..255 и заворачивает
        # в объект PIL.Image - такой можно показать или сохранить
        image_tensor = self.models.decode_latents(latents.float())
        return tensor_to_pil(image_tensor)
