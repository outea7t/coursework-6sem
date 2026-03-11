# classifier-free guidance
# eps_guided = eps_uncond + w * (eps_cond - eps_uncond)

import torch
from torch import Tensor


class ClassifierFreeGuidance:

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

        # батчинг uncond + cond в один forward pass
        latent_input = torch.cat([x, x])

        if t.dim() == 0:
            t_input = t.unsqueeze(0).expand(2)
        else:
            t_input = torch.cat([t, t])

        encoder_states = torch.cat([uncond_embeddings, cond_embeddings])

        added_cond_kwargs = {
            "text_embeds": torch.cat([uncond_pooled, cond_pooled]),
            "time_ids": torch.cat([time_ids, time_ids]),
        }

        noise_pred = model(
            latent_input,
            t_input,
            encoder_hidden_states=encoder_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        noise_uncond, noise_cond = noise_pred.chunk(2)

        noise_guided = noise_uncond + self.guidance_scale * (
            noise_cond - noise_uncond
        )

        return noise_guided
