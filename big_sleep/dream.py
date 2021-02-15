from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm, trange

from big_sleep import BigSleep
from big_sleep.big_sleep import perceptor, terminate
from big_sleep.clip import tokenize
from big_sleep.util import exists, open_folder


class Dream(nn.Module):
    def __init__(
            self,
            text,
            *,
            lr=.07,
            image_size=512,
            gradient_accumulate_every=1,
            save_every=50,
            epochs=20,
            iterations=1050,
            save_progress=False,
            bilinear=False,
            open_folder=True,
            seed=None,
            torch_deterministic=False,
            max_classes=None,
            class_temperature=2.,
            save_date_time=False,
            save_best=False,
            experimental_resample=False,
    ):
        super().__init__()

        if torch_deterministic:
            assert not bilinear, 'the deterministic (seeded) operation does not work with interpolation (PyTorch 1.7.1)'
            torch.set_deterministic(True)

        if exists(seed):
            print(f'setting seed of {seed}')
            if seed == 0:
                print(
                    'you can override this with --seed argument in the command line, or --random for a randomly chosen one')
            torch.manual_seed(seed)

        self.epochs = epochs
        self.iterations = iterations

        model = BigSleep(
            image_size=image_size,
            bilinear=bilinear,
            max_classes=max_classes,
            class_temperature=class_temperature,
            experimental_resample=experimental_resample,
        ).cuda(0)
        self.model = nn.DataParallel(model)
        self.lr = lr
        self.optimizer = Adam(self.model.module.model.latents.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.save_progress = save_progress
        self.save_date_time = save_date_time
        self.save_best = save_best
        self.current_best_score = 0
        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every
        self.set_text(text)

    def set_text(self, text):
        self.text = text
        textpath = self.text.replace(' ', '_')[:255]
        if self.save_date_time:
            textpath = datetime.now().strftime("%y%m%d-%H%M%S-") + textpath

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        encoded_text = tokenize(text).cuda()
        self.encoded_text = perceptor.encode_text(encoded_text).detach()

    def reset(self):
        self.model.reset()
        self.model = self.model.cuda()
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            losses = self.model(self.encoded_text)
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                top_score, best = torch.topk(losses[2], k=1, largest=False)
                image = self.model.module.model()[best].cpu()

                save_image(image, str(self.filename))
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f'image updated at "./{str(self.filename)}"')

                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    num = total_iterations // self.save_every
                    save_image(image, Path(f'./{self.textpath}.{num}.png'))

                if self.save_best and top_score.item() < self.current_best_score:
                    self.current_best_score = top_score.item()
                    save_image(image, Path(f'./{self.textpath}.best.png'))

        return total_loss

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        self.model(self.encoded_text)  # one warmup step due to issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        image_pbar = tqdm(total=self.total_image_updates, desc='image update', position=2, leave=True)
        for epoch in trange(self.epochs, desc='      epochs', position=0, leave=True):
            pbar = trange(self.iterations, desc='   iteration', position=1, leave=True)
            image_pbar.update(0)
            for i in pbar:
                loss = self.train_step(epoch, i, image_pbar)
                pbar.set_description(f'loss: {loss.item():04.2f}')

                if terminate:
                    print('detecting keyboard interrupt, gracefully exiting')
                    return