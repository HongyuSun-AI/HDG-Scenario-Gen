from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


class RNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, num_layers=2, use_gru=True):
        super(RNNFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gru = use_gru

        if use_gru:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)


        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        if not self.use_gru:
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)


        out = out[:, -1, :]
        out = self.fc(out)  # (batch_size, hidden_dim)
        return out


def augment_trajectory(data, noise_std=0.01):

    noise = torch.randn_like(data) * noise_std
    return data + noise


def contrastive_loss(features, temperature=0.5):

    batch_size = features.shape[0] // 2
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)


    labels = torch.arange(batch_size, device=features.device)
    labels = torch.cat([labels, labels], dim=0)


    logits = similarity_matrix / temperature
    loss = F.cross_entropy(logits, labels)

    return loss


class ModifiedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]

        # (C, H, W) -> (H*W, C)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2]).permute(1, 0)


        x_aug = augment_trajectory(x)

        return x, x_aug


def train_rnn_contrastive(model, dataloader, num_epochs=100, lr=0.001, temperature=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, x_aug in dataloader:
            x = x.to(device)
            x_aug = x_aug.to(device)

            features = model(torch.cat([x, x_aug], dim=0))  # (2N, feature_dim)
            loss = contrastive_loss(features, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    print("Contrastive Learning Training Complete.")

    base_dir = Path(__file__).parent.resolve()
    save_path = base_dir / 'deepcluster_model' / 'rnn_feature_extractor.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":

    num_samples = 1000
    seq_len = 140
    feature_dim = 36


    base_dir = Path(__file__).parent.resolve()
    load_path = base_dir / "processed_data" / "Track_dataset_smooth.pth"
    dataset = torch.load(load_path)


    load_path = (Path(__file__).parent / "processed_data" / "Track_dataset_smooth.pth").resolve()
    dataset = torch.load(load_path)


    dataset = ModifiedDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    feature_extractor = RNNFeatureExtractor(input_dim=feature_dim, hidden_dim=128, use_gru=True)
    train_rnn_contrastive(feature_extractor, dataloader, num_epochs=100, lr=1e-3, temperature=0.5)
