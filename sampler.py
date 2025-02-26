import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from model import Denoiser
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "GTZAN_dataset"
dataset = AudioDataset(path, train=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
model = torch.load('denoiser.pth', weights_only=False)

# metriky

n_steps = 10

model.eval()

for step in range(n_steps):
    for index, input_sig in enumerate(dataloader):
        with torch.no_grad():
            waveform, sample_rate = input_sig

            noise = 10
            corrupted_sig = waveform + noise * torch.randn(waveform.shape)

