import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from environment import BlackjackEnv 
from replay_buffer import ReplayBuffer
from hyperparameters import HyperParameters

"""
Dueling DQN architecture for the blackjack agent. This architecture separately estimates
the state value and action advantages, which is particularly useful for card games.
"""

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, x, training=True):
        # Simplified forward pass without batch norm handling
        x = self.feature_layer(x)
        
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.DEVICE = HyperParameters.DEVICE
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(HyperParameters.MEMORY_SIZE)
        self.gamma = HyperParameters.GAMMA
        self.epsilon = HyperParameters.EPSILON_START
        self.epsilon_min = HyperParameters.EPSILON_MIN
        self.epsilon_decay = HyperParameters.EPSILON_DECAY

        self.policy_net = DuelingDQN(state_size, action_size).to(self.DEVICE)
        self.target_net = DuelingDQN(state_size, action_size).to(self.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=HyperParameters.LEARNING_RATE)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, valid_actions=None):
        DEVICE = self.DEVICE
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            act_values = self.policy_net(state_tensor, training=False).cpu().data.numpy()

        # Filter out invalid actions by setting their values to a very low number
        masked_values = np.copy(act_values[0])
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_values[i] = -np.inf

        return np.argmax(masked_values)

    def learn(self):
        BATCH_SIZE = HyperParameters.BATCH_SIZE
        DEVICE = self.DEVICE
        
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states, training=True).gather(1, actions.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states, training=True).max(1)[0].detach()

        # Compute the expected Q values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
    
