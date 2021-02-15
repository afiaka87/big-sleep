import torch

from big_sleep.util import exists, differentiable_topk


class Latents(torch.nn.Module):
    def __init__(
            self,
            num_latents=32,
            max_classes=None,
            class_temperature=2.
    ):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, 128).normal_(std=1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, 1000).normal_(mean=-3.9, std=.3))
        self.register_buffer('thresh_lat', torch.tensor(1))

        assert not exists(
            max_classes) or 0 < max_classes <= 1000, 'num classes must be between 0 and 1000'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature=self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes