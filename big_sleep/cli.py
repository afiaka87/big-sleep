import random as rnd

import fire
import torch
from torch import nn
from big_sleep import Imagine
from .version import __version__


def train(
        text,
        lr=.07,
        image_size=512,
        gradient_accumulate_every=1,
        epochs=20,
        iterations=1050,
    save_every = 50,
    overwrite = False,
    save_progress = False,
    save_date_time = False,
    bilinear = False,
    open_folder = True,
    seed = 0,
    random = False,
    torch_deterministic = False,
    max_classes = None,
    class_temperature = 2.,
    save_best = False,
    experimental_resample = False,
):
    print(f'Starting up... v{__version__}')

    if random:
        seed = rnd.randint(0, 1e6)

    imagine = Imagine(
        text,
        lr = lr,
        image_size = image_size,
        gradient_accumulate_every = gradient_accumulate_every,
        epochs = epochs,
        iterations = iterations,
        save_every = save_every,
        save_progress = save_progress,
        bilinear = bilinear,
        seed = seed,
        torch_deterministic = torch_deterministic,
        open_folder = open_folder,
        max_classes = max_classes,
        class_temperature = class_temperature,
        save_date_time = save_date_time,
        save_best = save_best,
        experimental_resample = experimental_resample,
    ).cuda(0)
    if torch.cuda.device_count() > 1:
        imagine = nn.DataParallel(imagine)

    if not overwrite and imagine.filename.exists():
        answer = input('Imagined image already exists, do you want to overwrite? (y/n) ').lower()
        if answer not in ('yes', 'y'):
            exit()

    imagine()

def main():
    fire.Fire(train)
