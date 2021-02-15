from tqdm import trange

import os
import signal
from big_sleep import Dream
import torch
from torch import nn

phrase_one = "And blood-black nothingness began to spin a system of cells interlinked"  # @param {type:"string"}
phrase_two = "A system of cells interlinked within cells interlinked within cells interlinked within one stem"  # @param {type:"string"}
phrase_three = "And dreadfully distinct against the dark a tall white fountain played"  # @param {type:"string"}
phrase_four = "A system of cells interlinked within cells interlinked within cells interlinked within one stem"  # @param {type:"string"}
user_provided_phrases = [phrase_one, phrase_two, phrase_three, phrase_four]

texts = [non_empty_phrase for non_empty_phrase in user_provided_phrases if non_empty_phrase is not ""]
tmp_path = "/root/big-sleep-pycharm/output"

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

terminate = False


def signal_handling(signum, frame):
    global terminate
    terminate = True


signal.signal(signal.SIGINT, signal_handling)

lr = .07  # @param {type:"number"}
image_size = 512  # @param {type:"integer"}
iterations = 300  # @param {type:"integer"}
save_every = 1  # @param {type:"integer"}
overwrite = True  # @param {type:"boolean"}
save_progress = False  # @param {type:"boolean"}
save_date_time = True  # @param {type:"boolean"}
bilinear = False  # @param {type:"boolean"}
open_folder = True  # @param {type:"boolean"}
seed = 0  # @param {type:"integer"}
random = True  # @param {type:"boolean"}
torch_deterministic = False  # @param {type:"boolean"}
max_classes = None  # @param {type:"raw"}
class_temperature = 2.  # @param {type:"number"}
save_best = True  # @param {type:"boolean"}
experimental_resample = False  # @param {type:"boolean"}

epochs = len(texts)
gradient_accumulate_every = 1
text = texts[0]

model = Dream(text, lr=lr, image_size=image_size, gradient_accumulate_every=gradient_accumulate_every, epochs=epochs,
              iterations=iterations, save_every=save_every, save_progress=save_progress, bilinear=bilinear, seed=seed,
              torch_deterministic=torch_deterministic, open_folder=open_folder, max_classes=max_classes,
              class_temperature=class_temperature, save_date_time=save_date_time,
              save_best=save_best, experimental_resample=experimental_resample,
              ).cuda(0)
model = nn.DataParallel(model)

torch.cuda.empty_cache()
# In order per epoch phrase training
for epoch in trange(epochs, desc='epochs'):
    iter_pbar = trange(iterations, desc='iteration')
    text = texts[epoch]
    model.module.set_text(text)
    torch.cuda.empty_cache()
    for i in iter_pbar:
        loss = model.module.train_step(epoch, i)
        iter_pbar.set_description(f"loss: {loss}")
        if terminate:
            print('detecting keyboard interrupt, gracefully exiting')
            os.sys.exit()
