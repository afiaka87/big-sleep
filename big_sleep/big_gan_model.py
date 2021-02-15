from torch import nn

from big_sleep.biggan import BigGAN
from big_sleep.latents import Latents


class BigGanModel(nn.Module):
    def __init__(
            self,
            image_size,
            max_classes=None,
            class_temperature=2.
    ):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')

        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.init_latents()

    def init_latents(self):
        # noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
        self.latents = Latents(
            max_classes=self.max_classes,
            class_temperature=self.class_temperature
        )

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2