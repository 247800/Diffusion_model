# polished version of model.py

import torch
from torch import nn

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # if we don't mind the original tensor getting overwritten, we may save some memory by setting: nn.ReLU(inplace=True)
            # nn.MaxPool2d((2,1))
        )
        # self.bottleneck = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
