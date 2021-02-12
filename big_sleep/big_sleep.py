import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm, trange

from big_sleep.biggan import BigGAN
from big_sleep.clip import load, tokenize, normalize_image
from .resample import resample

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt

terminate = False


def signal_handling(signum, frame):
    global terminate
    terminate = True


signal.signal(signal.SIGINT, signal_handling)


# helpers

def exists(val):
    return val is not None


def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/', '\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass


# tensor helpers

def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x = x.scatter(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim=1)


# load clip

perceptor, preprocess = load()


# load biggan

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
            max_classes) or max_classes > 0 and max_classes <= 1000, 'num classes must be between 0 and 1000'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature=self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes


class Model(nn.Module):
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
        self.latents = Latents(
            max_classes=self.max_classes,
            class_temperature=self.class_temperature
        )

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2


# load siren

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

        self.model = Model(
            image_size=image_size,
            max_classes=max_classes,
            class_temperature=class_temperature
        )

    def reset(self):
        self.model.init_latents()

    def forward(self, max_text_tokenized: [], min_text_tokenized: [], return_loss=True):
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
            if (self.experimental_resample):
                apper = resample(apper, (224, 224))
            else:
                apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = normalize_image(into)

        image_embed = perceptor.encode_image(into)
        text_embeds = []
        for tokenized in max_text_tokenized:
            text_embed = perceptor.encode_text(tokenized.cuda()).detach().clone()
            text_embeds.append(text_embed)
        if min_text_tokenized is not None:
            min_text_embeds = []
            for tokenized in min_text_tokenized:
                text_embed = perceptor.encode_text(tokenized.cuda()).detach().clone()
                min_text_embeds.append(text_embed)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.thresh_lat

        latent_loss = torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                      torch.abs(torch.mean(latents)).mean() + \
                      4 * torch.max(torch.square(latents).mean(), latent_thres)

        for latent_arr in latents:
            mean = torch.mean(latent_arr)
            diffs = latent_arr - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

        latent_loss = latent_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents
        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest=False, dim=1, k=999)[0]) ** 2).mean()

        results = []
        for text_embed in text_embeds:
            results.append(self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean())
        if min_text_embeds is not None:
            for text_embed in min_text_embeds:
                results.append(-self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean())
        results = sum(results)
        return latent_loss, cls_loss, results


class Imagine(nn.Module):
    def __init__(
            self,
            text,
            penalty_text,
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
        ).cuda()

        self.model = model

        self.lr = lr
        self.optimizer = Adam(model.model.latents.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every

        text_max_tokenized, text_min_tokenized = self.split_and_tokenize_texts(text, penalty_text)
        self.set_text(text)
        self.text_min_tokenized = text_min_tokenized
        self.text_max_tokenized = text_max_tokenized

    def set_text(self, text):
        self.text = text
        textpath = self.text.replace(' ', '_')[:255]
        if self.save_date_time:
            textpath = datetime.now().strftime("%y%m%d-%H%M%S-") + textpath

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        #  TODO allow user to set both max and min here
        self.text_max_tokenized, _ = self.split_and_tokenize_texts(text)

    def split_and_tokenize_texts(self, text_to_max: str, text_to_min: str = None):
        texts_to_max_tokenized = []
        if "\\" in text_to_max:
            for prompt in text_to_max.split("\\"):
                texts_to_max_tokenized.append(tokenize(f'''{prompt}'''))
        else:
            texts_to_max_tokenized = tokenize(text_to_max)
        texts_to_min_tokenized = []
        if text_to_min is not None:
            if "\\" in text_to_max:
                for prompt in text_to_min.split("\\"):
                    texts_to_min_tokenized.append(tokenize(f'''{prompt}'''))
            else:
                texts_to_max_tokenized = tokenize(text_to_min)
        return texts_to_max_tokenized, texts_to_min_tokenized

    def reset(self):
        self.model.reset()
        self.model = self.model.cuda()
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            if self.text_min_tokenized is None:
                losses = self.model(self.text_max_tokenized)
            else:
                losses = self.model(self.text_max_tokenized, self.text_min_tokenized)
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                top_score, best = torch.topk(losses[2], k=1, largest=False)
                image = self.model.model()[best].cpu()

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

        self.model(self.text_max_tokenized, self.text_min_tokenized)  # one warmup step due to issue with CLIP and CUDA

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
