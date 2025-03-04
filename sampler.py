import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from model import Denoiser
import torchvision.models as models
import auraloss

def get_time_schedule(sigma_min=1e-5, sigma_max=12, T=50, rho=10):
    i = torch.arange(0, T + 1)
    # N = torch.randn(1)
    sigma_i = (sigma_max ** (1/rho) + i * (sigma_min ** (1/rho) - sigma_max ** (1/rho)) / (T - 1)) ** rho
    sigma_i[T] = 0
    return sigma_i

def get_noise(t, S_tmin=1e-5, S_tmax=12, S_churn=None, idx=0):
    if S_churn is None:
        S_churn = t[idx:idx+1]
        print('S_churn:', S_churn)
    if S_churn < S_tmin:
        gamma_i = 0
    elif S_churn > S_tmax:
        gamma_i = 0
    else:
        N = torch.randn(1)
        gamma_i = min(S_churn/N, np.sqrt(1) - 1)
    return gamma_i

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "GTZAN_dataset"
dataset = AudioDataset(path, train=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
model = Denoiser()
model.load_state_dict(torch.load('denoiser.pt', weights_only=True))

loss_func = auraloss.freq.MultiResolutionSTFTLoss()
test_loss = []
n_steps = 10

S_noise = 1
t = get_time_schedule()

model.eval()

for step in range(n_steps):
    for index, input_sig in enumerate(dataloader):
        with torch.no_grad():
            x, _ = input_sig
            x_0 = x + torch.randn(x.shape)
            
            gamma = get_noise(t=t, idx=index)
            t_i = t + t * gamma
            
            # corrupted_sig = waveform + noise * torch.randn(waveform.shape)
            # denoised_sig = corrupted_sig.squeeze(0).squeeze(0)
            # sample = model(denoised_sig.unsqueeze(0).unsqueeze(0))

            # test_loss.append(loss.item())


