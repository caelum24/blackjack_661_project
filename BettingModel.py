import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np


class BettingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y
    

class BettingModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super(BettingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # final output, e.g. bet amount
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    

def train_betting_model(
    model,
    train_loader,
    val_loader=None,
    lr=1e-3,
    epochs=20,
    device="cpu"
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(batch_features)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        if val_loader is not None:
            model.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    epoch_val_loss += loss.item() * len(batch_features)

            epoch_val_loss /= len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            print(f"Epoch: {epoch+1:02d}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        else:
            print(f"Epoch: {epoch+1:02d}, Train Loss: {epoch_train_loss:.4f}")

    return train_losses, val_losses
