import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from model import Denoiser
import utils.sampling_utils as s_utils
import torchvision.models as models
import auraloss
import wandb

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "GTZAN_dataset"
dataset = AudioDataset(path, train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
model = Denoiser()
model.load_state_dict(torch.load('denoiser.pt', weights_only=True))

loss_func = auraloss.freq.MultiResolutionSTFTLoss()
# test_loss = []
n_steps = 50

S_noise = 1
t = s_utils.get_time_schedule()

model.eval()

learning_rate = 0.001
n_epochs = 5
run = wandb.init(
    project="dnn-sampler",
    config={
        "learning_rate": learning_rate,
        "epochs": n_epochs,
    },
)

input_sig = next(iter(dataloader))
x = input_sig
x_0 = x + torch.randn(x.shape)

for step in range(n_steps):
        with torch.no_grad():
            gamma = s_utils.get_noise(t=t, idx=step)
            t_hat = t[step] + t[step] * gamma
            noise = torch.randn(x.shape)
            x_hat = x + torch.sqrt(t_hat.clone().detach() ** 2 - t[step] ** 2) * noise
            D_theta = model(x_hat)
            d = (x_hat - D_theta) / t_hat
            x = x_hat + (t[step + 1] - t_hat) * d
            loss = loss_func(x_i, x).to(device)
            # print("step:", step)
            print(f"Step: [{step+1}/{n_steps}], Loss: {loss}")
            wandb.log({"loss": loss})
            # test_loss.append(loss.item())

wandb.finish()

