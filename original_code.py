promptInputMaximize = "hyperrealistic dream landscape"  # @param{type:"string"}
promptInputMinimize = ""  # @param{type:"string"}
displ_freq = 25  # @param{type:"number"}
save_imgs = False  # @param{type:"boolean"}

im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape

txMs = []
if promptInputMaximize != "":
    for prompt in promptInputMaximize.split("\\"):
        txMs.append(clip.tokenize(f'''{prompt}'''))

txIs = []
if promptInputMinimize != "":
    for prompt in promptInputMinimize.split("\\"):
        txIs.append(clip.tokenize(f'''{prompt}'''))


# LATENT COORDINATE DEFINES

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        self.normu = torch.nn.Parameter(torch.zeros(32, 128).normal_(std=1).cuda())
        params_other = torch.zeros(32, 1000).normal_(-3.9, .3)
        self.cls = torch.nn.Parameter(params_other)
        self.thrsh_lat = torch.tensor(1).cuda()
        self.thrsh_cls = torch.tensor(1.9).cuda()

    def forward(self):
        return self.normu, torch.sigmoid(self.cls)


# LATENT COORDINATE FUNCTIONS

torch.manual_seed(0)

lats = Pars().cuda()
print("NORMU")
print(lats.normu.shape)
print("CLS")
print(lats.cls.shape)
optimizer = torch.optim.Adam(lats.parameters(), .07)
eps = 0

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

tMs = []
for txM in txMs:
    tMs.append(perceptor.encode_text(txM.cuda()).detach().clone())

tIs = []
for txI in txIs:
    tIs.append(perceptor.encode_text(txI.cuda()).detach().clone())

with torch.no_grad():
    al = (model(*lats(), 1).cpu()).numpy()

import datetime

if save_imgs:
    now = datetime.datetime.now()
    nowStr = now.strftime("%Y-%m-%d %H-%M-%S")
    folderName = f"{prompt}-progress {nowStr}"
    savePath = os.path.join(drivePath, folderName)
    !mkdir
    "$savePath"

itt = 0


def checkin(loss):
    print(loss)
    best = torch.topk(loss[2], k=1, largest=False)[1]
    with torch.no_grad():
        al = model(*lats(), 1)[best:best + 1].cpu().numpy()
    for allls in al:
        if save_imgs:
            displSample(allls, os.path.join(savePath, f"{itt:05}.png"))
        else:
            displSample(allls, "")
        display.display(display.Image(str(3) + '.png'))
        print('\n')


def ascend_txt():
    out = model(*lats(), 1)

    cutn = 128
    p_s = []
    for ch in range(cutn):
        size = int(sideX * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(.5, .95))
        offsetx = torch.randint(0, sideX - size, ())
        offsety = torch.randint(0, sideX - size, ())
        apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(apper, (224, 224), mode='nearest')
        p_s.append(apper)
    into = torch.cat(p_s, 0)

    into = nom((into + 1) / 2)
    iii = perceptor.encode_image(into)
    llls = lats()
    lat_l = torch.abs(1 - torch.std(llls[0], dim=1)).mean() + \
            torch.abs(torch.mean(llls[0])).mean() + \
            4 * torch.max(torch.square(llls[0]).mean(), lats.thrsh_lat)

    for array in llls[0]:
        mean = torch.mean(array)
        diffs = array - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
        lat_l = lat_l + torch.abs(kurtoses) / llls[0].shape[0] + torch.abs(skews) / llls[0].shape[0]

    cls_l = ((50 * torch.topk(llls[1], largest=False, dim=1, k=999)[0]) ** 2).mean()
    result = [lat_l, cls_l]

    for t in tMs:
        result.append(-100 * torch.cosine_similarity(t, iii, dim=-1).mean())
    for t in tIs:
        result.append(100 * torch.cosine_similarity(t, iii, dim=-1).mean())
    return result


def train(epoch, i):
    lossAll = ascend_txt()
    loss = sum(lossAll)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if itt % displ_freq == 0:
        checkin(lossAll)


itt = 0
for epochs in range(10000):
    for i in range(50000):
        train(eps, i)
        itt += 1
    eps += 1
