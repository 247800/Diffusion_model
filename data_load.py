import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Audio_Dataset(Dataset):
    def __init__(self, annotations, directory):
        self.annotations = pd.read_csv(annotations)
        self.directory = directory

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = os.path.join(self.directory, self.annotations.iloc[index, 0])
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate