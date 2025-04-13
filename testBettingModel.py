import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np


class BettingDataset(Dataset):
    """
    A Dataset class for supervised learning of bet sizes.
    Expected data format (adapt to your project):
      - features: array-like of shape (num_samples, num_features)
                   e.g. [card_count, true_count, etc...]
      - targets: array-like of shape (num_samples,)
                 e.g. bet size or classification label
    """
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
    """
    Example neural network for predicting an optimal bet 
    (or some bet-related target). You can adapt architecture
    and activation to your needs.
    """
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
    """
    Train the BettingModel using MSE or another suitable loss.
    """
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


def plot_basic_metrics(bankroll_history, loss_history, reward_history):
    """
    Plots basic metrics:
    - Bankroll over time
    - Loss over epochs
    - Reward over episodes
    """
    # 1) Bankroll
    plt.figure()
    plt.plot(bankroll_history)
    plt.title("Bankroll Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll")
    plt.show()

    # 2) Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 3) Reward
    plt.figure()
    plt.plot(reward_history)
    plt.title("Reward Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Reward")
    plt.show()


def plot_epsilon_bonus_decay(epsilon_values, bonus_values):
    """
    Plot how epsilon and bonus change over training steps or episodes.
    """
    # 1) Epsilon decay
    plt.figure()
    plt.plot(epsilon_values)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Epsilon")
    plt.show()

    # 2) Bonus decay
    plt.figure()
    plt.plot(bonus_values)
    plt.title("Bonus Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bonus")
    plt.show()


def plot_hyperparameter_results(hparam_results):
    """
    Plot hyperparameter search results (e.g. best reward vs. hyperparameters).
    hparam_results could be a dictionary or list of (param, performance).
    Example structure:
      hparam_results = {
        'lr=1e-3, hidden=64': 0.56,
        'lr=1e-4, hidden=128': 0.58,
        ...
      }
    """
    labels = list(hparam_results.keys())
    values = list(hparam_results.values())

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Hyperparameter Search Results")
    plt.xlabel("Hyperparameter Setting")
    plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.show()


def plot_comparative_strategies(basic_strategy_results, learned_strategy_results, counting_strategy_results=None):
    """
    Compare different strategies (basic, learned, counting).
    Each input might be a list of rewards or bankroll over time.
    """
    plt.figure()
    plt.plot(basic_strategy_results, label="Basic Strategy")
    plt.plot(learned_strategy_results, label="Learned Strategy")
    
    if counting_strategy_results is not None:
        plt.plot(counting_strategy_results, label="Counting Strategy")
    
    plt.title("Comparative Strategies")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll or Reward")
    plt.legend()
    plt.show()


def plot_action_values(action_values):
    """
    Shows a histogram (or other chart) of the relative values
    of each action in a given state (or aggregated states).
    
    `action_values` might be a dict or list of Q-values or predicted values:
      {
        'Hit': 0.12,
        'Stand': -0.02,
        'Double': 0.08,
        ...
      }
    """
    actions = list(action_values.keys())
    values = list(action_values.values())

    plt.figure()
    plt.bar(actions, values)
    plt.title("Relative Value of Each Action")
    plt.xlabel("Action")
    plt.ylabel("Estimated Value")
    plt.show()
