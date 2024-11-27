import torch
from torch import nn

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # self.bottleneck = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
        )

    def forward(self,x):
        x = self.encoder(x)
        # x = self.bottleneck(x)
        x = self.decoder(x)
        return x
