import torch
import torchaudio
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import AudioDataset
from model import Denoiser
import auraloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)

path = "GTZAN_dataset"
dataset = AudioDataset(path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = Denoiser()

# loss_func = nn.MSELoss()
loss_func = auraloss.freq.MultiResolutionSTFTLoss()

learning_rate = 0.001
optimizer = optim.Adam(model_train.parameters(), lr=learning_rate)

n_epochs = 5
epoch = 0
train_loss = []

model.train()

for epoch in range(n_epochs):
    for index, input_sig in enumerate(dataloader):
        waveform, sample_rate = input_sig
        noise = 10
        corrupted_sig = waveform + noise * torch.randn(waveform.shape)

        optimizer.zero_grad()
        output = model_train(corrupted_sig)
        loss = loss_func(output, waveform).to(device)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        print(f"Epoch: [{epoch+1}/{n_epochs}], Step: {index+1}, Loss: {loss}")
        # print("Loss device:", loss.device)
        # print(f"End of epoch {epoch}, accuracy {acc}")

torch.save(model.state_dict(), 'denoiser.pt')
