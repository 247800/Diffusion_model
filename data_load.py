import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, directory):
        # self.annotations = pd.read_csv(annotations)
        self.directory = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".wav")]

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, index):
        path = self.directory[index]
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate
