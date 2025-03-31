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



class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
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

        self.policy_net = DQN(state_size, action_size).to(self.DEVICE)
        self.target_net = DQN(state_size, action_size).to(self.DEVICE)
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
            act_values = self.policy_net(state_tensor).cpu().data.numpy()

        # Filter out invalid actions by setting their values to a very low number
        masked_values = np.copy(act_values[0])
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_values[i] = -np.inf

        return np.argmax(masked_values)

    def bet(self, state):
        # Added Kelly Criterion to for the bet factor
        win_prob = 0.3610
        loss_prob = 0.5200
        net_odds = 1.5

        kelly_fraction = (net_odds * win_prob - loss_prob) / net_odds

        kelly_fraction = min(1, max(kelly_fraction, 0))

        # Using a separate neural network for betting would be better,
        #if np.random.rand() <= self.epsilon:
            #return np.random.uniform(0, 1)  # Random bet size factor between 0 and 1

        # For exploration during training, add some noise to the bet
        #noise = 0.1 * np.random.randn()
        #bet_factor = state[5] + noise  # Use the previous bet as a starting point
        #return max(0, min(1, bet_factor))  # Clip to [0, 1]
        return kelly_fraction + 0.01 * np.random.randn()


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
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states).max(1)[0].detach()

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