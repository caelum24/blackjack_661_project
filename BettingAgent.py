import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from replay_buffer import ReplayBuffer

class BettingNet(nn.Module):
    def __init__(self, state_dim, num_bets, hidden_dim=64):
        super(BettingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bets)
        )

    def forward(self, x):
        return self.net(x)


class BettingRLAgent:
    def __init__(
        self,
        state_dim,
        possible_bets, #discrete bets for DQN approach
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1e-5,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000,
        device="cpu"
    ):
        self.state_dim = state_dim
        self.possible_bets = possible_bets
        self.num_bets = len(possible_bets)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        self.target_update_freq = target_update_freq

        self.policy_net = BettingNet(state_dim, self.num_bets).to(device)
        self.target_net = BettingNet(state_dim, self.num_bets).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0

    def act(self, state):
        self.decay_epsilon()

        if random.random() < self.epsilon:
            bet_index = random.randrange(self.num_bets)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            bet_index = q_values.argmax(dim=1).item()

        return self.possible_bets[bet_index]

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

    def store_transition(self, state, action_idx, reward, next_state, done):
        self.memory.push(state, action_idx, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Current Q
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]       # shape: (batch_size, num_bets)

        # Bellman target
        targets = rewards + (1 - dones.float()) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

        # Update target network periodically
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.decay_epsilon()

        return loss.item()