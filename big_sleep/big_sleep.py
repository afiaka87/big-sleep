import signal

import torch
import torch.nn.functional as F
from torch import nn

from big_sleep.clip import load, normalize_image
from .big_gan_model import BigGanModel
from .resample import resample
from .util import signal_handling

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt
terminate = False
signal.signal(signal.SIGINT, signal_handling)

perceptor, preprocess = load()


class BigSleep(nn.Module):
    def __init__(
            self,
            num_cutouts=128,
            loss_coef=100,
            image_size=512,
            bilinear=False,
            max_classes=None,
            class_temperature=2.,
            experimental_resample=False,
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.experimental_resample = experimental_resample

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.model = BigGanModel(
            image_size=image_size,
            max_classes=max_classes,
            class_temperature=class_temperature
        )

    def reset(self):
        self.model.init_latents()

    def forward(self, text_embed, return_loss=True):
        width, num_cutouts = self.image_size, self.num_cutouts

        out = self.model()

        if not return_loss:
            return out

        pieces = []
        for ch in range(num_cutouts):
            size = int(width * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
            offsetx = torch.randint(0, width - size, ())
            offsety = torch.randint(0, width - size, ())
            apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
            if self.experimental_resample:
                apper = resample(apper, (224, 224))
            else:
                apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = normalize_image(into)

        image_embed = perceptor.encode_image(into)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.thresh_lat

        lat_loss = torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                   torch.abs(torch.mean(latents, dim=1)).mean() + \
                   4 * torch.max(torch.square(latents).mean(), latent_thres)

        for array in latents:
            mean = torch.mean(array)
            diffs = array - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

            lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents

        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest=False, dim=1, k=999)[0]) ** 2).mean()

        sim_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        return lat_loss, cls_loss, sim_loss
