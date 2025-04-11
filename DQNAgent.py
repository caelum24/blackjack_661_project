import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
# from environment import BlackjackEnv 
from split_environment import BlackjackEnv
from replay_buffer import ReplayBuffer
from hyperparameters import HyperParameters
from epsilon_decayer import EpsilonDecayer

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


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)


class DQNAgent:

    counting_type_state_size = {"full":15, "empty":5, "hi_lo":6, "zen":6, "uston_apc":6, "ten_count":6}

    def __init__(self, count_type):
        
        # TODO -> do we want to include the ability to exclude splitting?
        if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
            print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"])
            raise ValueError 

        state_size = self.counting_type_state_size[count_type]
        action_size = 4 # hit, stand, double, split

        self.DEVICE = HyperParameters.DEVICE
        self.count_type = count_type
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

    def remember(self, state, action, reward, next_state, split_state, done):
        self.memory.push(state, action, reward, next_state, split_state, done)

    def act(self, state, eval = False):

        # Determine valid actions to ensure model doesn't do something illegal
        valid_actions = [0, 1]  # Hit and stand are always valid
        if state[3] == 1:  # Can double
            valid_actions.append(2)
        if state[4] == 1:
            valid_actions.append(3)

        if not eval and np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            act_values = self.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

        # Filter out invalid actions by setting their values to a very low number
        masked_values = np.copy(act_values)
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_values[i] = -np.inf

        return np.argmax(masked_values)

    def bet(self, state):
        #TODO -> this can go away because we're going to have a separate agent for betting
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
        
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states_1, split_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.DEVICE)
        actions = torch.LongTensor(actions).to(self.DEVICE)
        rewards = torch.FloatTensor(rewards).to(self.DEVICE)
        next_states_1 = torch.FloatTensor(next_states_1).to(self.DEVICE)
        split_states = torch.FloatTensor(split_states).to(self.DEVICE)
        has_next2 = (split_states.abs().sum(dim=1) > 1e-5).float() # checking if next_states_2 is a 0 vector (only non-zero if a split occurred)
        dones = torch.FloatTensor(dones).to(self.DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states, training=True).gather(1, actions.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        # next_q_values = self.target_net(next_states_1, training=True).max(1)[0].detach()
        #TODO -> print out the outputs to see what they get out of it, why is our reward system diverging

        q_options1 = self.target_net(next_states_1)
        q_options2 = self.target_net(split_states)
        # print("STATES")
        # print(next_states_1, split_states)
        # print("Q_VALUES")
        # print(q_options1, q_options2)

        next_q_values1 = self.target_net(next_states_1).max(1)[0].detach()
        next_q_values2 = self.target_net(split_states).max(1)[0].detach()
        # print(next_q_values2*has_next2)
        # Compute the expected Q values
        # THIS IS THE 2 STATE STUFF FOR SPLITTING
        combined_next_q_values = next_q_values1 + has_next2 * next_q_values2
        # print("COMBINED Qs", combined_next_q_values)
        expected_q_values = rewards + (1 - dones) * self.gamma * combined_next_q_values
        # print("DATA", rewards, dones, combined_next_q_values)

        # print("COMBINED EXPECTED", expected_q_values)
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values1
        # print("EXPECTED", expected_q_values)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # TODO -> add epsilon decay class here to benefit training
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            pass

        return loss.item()
    
    def load_model(self, weight_file):
        self.target_net.load_state_dict(torch.load(weight_file))

    def save_model(self):
        model_checkpoint = "BJ_agent.pt"
        torch.save(self.target_net.state_dict(), model_checkpoint)
