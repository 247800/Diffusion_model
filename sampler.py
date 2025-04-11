import torch
import torchaudio
import torch.optim as optim
import numpy as np
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
# from model import Denoiser
from unet_octCQT import Unet_octCQT
import utils.cqt_nsgt_pytorch.CQT_nsgt
import utils.sampling_utils as s_utils
from utils.loss import l2_comp_stft_sum as loss_fn
# import auraloss
import wandb

import hydra
import os

from omegaconf import DictConfig
from omegaconf import ListConfig
import torch.serialization
from omegaconf.base import ContainerMetadata
from typing import Any
from collections import defaultdict
from omegaconf.nodes import AnyNode
from omegaconf.base import Metadata

torch.serialization.add_safe_globals([ListConfig])
torch.serialization.add_safe_globals([ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([AnyNode])
torch.serialization.add_safe_globals([Metadata])
torch.serialization.add_safe_globals([DictConfig])

os.environ['HYDRA_FULL_ERROR'] = '1'

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

hydra.initialize(config_path="config", version_base="1.4")  # Set correct path
cfg = hydra.compose(config_name="cqtdiff+_44k_32binsoct")  # Use correct config name
model = hydra.utils.instantiate(cfg)
model.load_state_dict(torch.load("checkpoints/guitar_Career_44k_6s-325000.pt", weights_only=True))

path = "GTZAN_dataset"
dataset = AudioDataset(path, train=False, seg_len=262144)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# model = Denoiser()
# model.load_state_dict(torch.load('denoiser.pt', weights_only=True))
# model = Unet_octCQT(
#     depth=8,
#     emb_dim=256,
#     Ns=[32,32, 64 ,64, 128, 128,256, 256],
#     attention_layers=[0, 0, 0, 0, 0, 0, 0, 0],
#     checkpointing=[True, True, True, True, True, True, True, True],
#     Ss=[2,2,2,2, 2, 2, 2, 2],
#     num_dils=[1,3,4,5,5,6,6,7],
#     cqt = {
#         "window": "kaiser",
#     	"beta": 1,
#     	"num_octs": 8,
# 	"bins_per_oct": 32,
#     },
#     bottleneck_type="res_dil_convs",
#     num_bottleneck_layers=1,
#     attention_dict = {
# 	"num_heads": 8,
#         "attn_dropout": 0.0,
#     	"bias_qkv": False,
# 	"N": 0,
#     	"rel_pos_num_buckets": 32,
#     	"rel_pos_max_distance": 64,
# 	"use_rel_pos": True,
#    	"Nproj": 8
#     }
# )

# loss_func = auraloss.freq.MultiResolutionSTFTLoss()
n_steps = 50
S_noise = 1
t = s_utils.get_time_schedule()

model.eval()
model.to(device)

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
x_i = input_sig.to(device)
# x_0 = x + torch.randn(x.shape)

for step in range(n_steps):
        with torch.no_grad():
            gamma = s_utils.get_noise(t=t, idx=step)
            t_hat = t[step] + t[step] * gamma
	    t_hat = t_hat.to(device)
            noise = torch.randn(x_i.shape, device=device)
            x_hat = x_i + torch.sqrt(t_hat.clone().detach() ** 2 - t[step] ** 2) * noise
            D_theta = model(x_hat, sigma=t[step])
            d = (x_hat - D_theta) / t_hat
            x_i = x_hat + (t[step + 1] - t_hat) * d
            loss = loss_fn(x=x.squeeze(1), x_hat=x_i.squeeze(1)).to(device)
            
            print(f"Step: [{step+1}/{n_steps}], Loss: {loss}")
            wandb.log({"loss": loss})
            # test_loss.append(loss.item())

wandb.finish()

