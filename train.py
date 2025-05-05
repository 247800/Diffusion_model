import torch
import torchaudio
import torch.optim as optim
from torch import nn
from dataset import AudioDataset
from torch.utils.data import Dataset, DataLoader
from model import Denoiser
from CQT.unet_octCQT import Unet_octCQT
import CQT.CQT_nsgt
# import auraloss
from utils.loss import l2_comp_stft_sum as loss_fn
import wandb

wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "GTZAN_dataset"
dataset = AudioDataset(path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = Denoiser()
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

# loss_func = nn.MSELoss()
# loss_func = auraloss.freq.MultiResolutionSTFTLoss()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
n_epochs = 50
epoch = 0
# train_loss = []

 run = wandb.init(
     project="dnn-train",
     config={
         "learning_rate": learning_rate,
         "epochs": n_epochs,
     },
 )

model.train()

for epoch in range(n_epochs):
    for index, input_sig in enumerate(dataloader):
        waveform, sample_rate = input_sig
        noise = 10
        corrupted_sig = waveform + noise * torch.randn(waveform.shape)

        optimizer.zero_grad()
        output = model(corrupted_sig)
        # loss = loss_func(output, waveform).to(device)
        loss = loss_fn(x=waveform.squeeze(1), x_hat=output.squeeze(1)).to(device)
        loss.backward()
        optimizer.step()
        # train_loss.append(loss.item())
	wandb.log({"loss": loss})

        print(f"Epoch: [{epoch+1}/{n_epochs}], Step: {index+1}, Loss: {loss}")
        # print("Loss device:", loss.device)
        # print(f"End of epoch {epoch}, accuracy {acc}")
        
# torch.save(model.state_dict(), 'denoiser.pt')
wandb.finish()
