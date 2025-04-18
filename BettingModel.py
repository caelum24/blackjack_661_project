import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BettingNN(nn.Module):
    def __init__(self, input_dim=3):
        super(BettingNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_betting_nn(dictionary, count_type, batch_size, epochs):

    counting_type_state_size = {"full":10, "hi_lo":1, "zen":1, "uston_apc":1, "ten_count":1, "comb_counts":4}

    if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count", "comb_counts"]:
        print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count", "comb_counts"])
        raise ValueError 
    state_size = counting_type_state_size[count_type]

    dataset = CountRewardDataset(dictionary)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    model = BettingNN(state_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model

class CountRewardDataset(Dataset):
    def __init__(self, count_reward_dict):
        self.inputs = []
        self.labels = []

        for count_state, (avg_reward, _) in count_reward_dict.items():
            input_tensor = torch.tensor(count_state, dtype=torch.float32)  # shape (3,)
            label_tensor = torch.tensor([avg_reward], dtype=torch.float32)  # shape (1,)
            self.inputs.append(input_tensor)
            self.labels.append(label_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]