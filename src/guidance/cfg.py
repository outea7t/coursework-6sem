# направление генерации без классификатора (Classifier-Free Guidance).
#
# приём, благодаря которому картинка реально слушается описания.
# на каждом шаге u-net вызывается два раза:
#   - с описанием (что мы хотим увидеть);
#   - без описания (или с негативным).
# потом два ответа смешиваем с коэффициентом w. чем больше w -
# тем сильнее картинка тянется к запросу. на практике обычно ставят w≈7.5.
#
# чтобы не звать сеть два раза подряд, я склеиваю обе версии входа
# в один батч и делаю один общий вызов - так быстрее.

import torch
from torch import Tensor


class ClassifierFreeGuidance:

    def __init__(self, guidance_scale: float = 7.5) -> None:
        # коэффициент усиления w. чем больше - тем сильнее картинка
        # тянется к описанию (но если перестараться - искажения)
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

        # частный случай: если коэффициент равен 1, то направление выключено -
        # достаточно одного вызова сети с условием
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

        # объединяем два варианта входа в один батч.
        # вместо двух отдельных вызовов u-net делаем один - так быстрее.
        # порядок: сначала uncond, потом cond
        latent_input = torch.cat([x, x])

        # повторяем шаг времени для обеих копий
        if t.dim() == 0:
            t_input = t.unsqueeze(0).expand(2)
        else:
            t_input = torch.cat([t, t])

        # склеиваем условия двух промптов
        encoder_states = torch.cat([uncond_embeddings, cond_embeddings])

        added_cond_kwargs = {
            "text_embeds": torch.cat([uncond_pooled, cond_pooled]),
            "time_ids": torch.cat([time_ids, time_ids]),
        }

        # один проход сети сразу для двух картинок
        noise_pred = model(
            latent_input,
            t_input,
            encoder_hidden_states=encoder_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # разрезаем результат обратно на две части
        noise_uncond, noise_cond = noise_pred.chunk(2)

        # смешиваем оба предсказания с коэффициентом усиления
        noise_guided = noise_uncond + self.guidance_scale * (
            noise_cond - noise_uncond
        )

        return noise_guided
