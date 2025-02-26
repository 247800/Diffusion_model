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
