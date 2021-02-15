from tqdm import trange

import os
import signal
from big_sleep import Dream
import torch
from torch import nn


texts = [
    "hollow knight hd png",
    "hollow knight stag hd png",
    "hollow knight dream hd png",
    "hollow knight git gud hd png",
    "hollow knight pixel hd png",
    "hollow knight pixel png hd",
    "hollow knight eyes hd png",
    "hollow knight pinterest hd png",
    "hollow knight sprite hd png",
    "hollow knight animation hd png",
    "hollow knight animation hollow knight character hd png",
    "hornet hollow knight characters hd png",

]
tmp_path = "/root/big-sleep-pycharm/output"
terminate = False


def mkdir_and_chdir(tmp_path):
    try:
        os.mkdir(tmp_path)
    except OSError:
        print("Creation of the directory %s failed" % tmp_path)
    else:
        print("Successfully created the directory %s " % tmp_path)

    try:
        os.chdir(tmp_path)
    except OSError:
        print("Changing into directory %s failed" % tmp_path)
    else:
        print("Successfully changed into directory %s" % tmp_path)



def signal_handling(signum, frame):
    global terminate
    terminate = True


signal.signal(signal.SIGINT, signal_handling)

lr = .065  # @param {type:"number"}
image_size = 512  # @param {type:"integer"}
epochs = 5 # @param {type:"integer"}
iterations = 200 # @param {type:"integer"}
save_every = 5  # @param {type:"integer"}
overwrite = True  # @param {type:"boolean"}
save_progress = True # @param {type:"boolean"}
save_date_time = False # @param {type:"boolean"}
bilinear = True # @param {type:"boolean"}
open_folder = True  # @param {type:"boolean"}
torch_deterministic = False  # @param {type:"boolean"}
max_classes = None # @param {type:"raw"}
class_temperature = 2.05  # @param {type:"number"}
save_best = True  # @param {type:"boolean"}
experimental_resample = True # @param {type:"boolean"}

clean_runs = len(texts)
gradient_accumulate_every = 1


for text in texts:
    spaceless_text = text.replace(" ", "_")
    mkdir_and_chdir(f"{tmp_path}/{spaceless_text}")
    model = Dream(text, lr=lr, image_size=image_size, gradient_accumulate_every=gradient_accumulate_every, epochs=epochs,
                  iterations=iterations, save_every=save_every, save_progress=save_progress, bilinear=bilinear,
                  torch_deterministic=torch_deterministic, open_folder=open_folder, max_classes=max_classes,
                  class_temperature=class_temperature, save_date_time=save_date_time,
                  save_best=save_best, experimental_resample=experimental_resample,
                  )
    # In order per epoch phrase training
    for epoch in trange(1, desc='epochs'):
        iter_pbar = trange(iterations, desc='iteration')
        text = texts[epoch]
        model.set_text(text)
        torch.cuda.empty_cache()
        for i in iter_pbar:
            loss = model.module.train_step(epoch, i)
            iter_pbar.set_description(f"loss: {loss}")
            if terminate:
                print('detecting keyboard interrupt, gracefully exiting')
                os.sys.exit()
        del model
        torch.cuda.empty_cache()