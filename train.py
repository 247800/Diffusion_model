import torch
import torchaudio
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data_load import AudioDataset
from model import Denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "dataset"
dataset = AudioDataset(path)
model_train = Denoiser()

loss_func = nn.MSELoss()

learning_rate = 0.001
optimizer = optim.Adam(model_train.parameters(), lr=learning_rate)

def train_loop():
    train_loss = 0
    loss = loss_func()
    train_loss += loss.item()
    return train_loss / len(dataset)
