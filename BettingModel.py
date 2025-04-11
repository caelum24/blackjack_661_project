import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        # args matches Transition namedtuple: (state, action, reward, next_state, done)
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool))

    def __len__(self):
        return len(self.buffer)


class BettingRLAgent:
    def __init__(
        self,
        state_dim,
        possible_bets,      # list or array of discrete bet sizes
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

        # Main network (policy)
        self.policy_net = BettingNet(state_dim, self.num_bets).to(device)
        # Target network
        self.target_net = BettingNet(state_dim, self.num_bets).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # For tracking training steps
        self.train_steps = 0

    def select_action(self, state):
        # Decay epsilon before selecting action
        self.decay_epsilon()

        if random.random() < self.epsilon:
            # Random action
            bet_index = random.randrange(self.num_bets)
        else:
            # Greedy w.r.t. Q-values
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            bet_index = q_values.argmax(dim=1).item()
        
        return bet_index

    def choose_bet_size(self, state):
        bet_index = self.select_action(state)
        bet_amount = self.possible_bets[bet_index]
        return bet_amount

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
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q
        q_values = self.policy_net(states)              # shape: (batch_size, num_bets)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # shape: (batch_size,)

        # Next Q from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)       # shape: (batch_size, num_bets)
            max_next_q_values = next_q_values.max(dim=1)[0]    # shape: (batch_size,)

        # Bellman target
        targets = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        # Update target network periodically
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_betting_agent():

    state_dim = 5     # e.g., [player_total, dealer_upcard, card_count, etc...]
    possible_bets = [1, 5, 10, 25, 50]  # discrete bet sizes
    agent = BettingRLAgent(
        state_dim=state_dim,
        possible_bets=possible_bets,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=1000,
        device="cpu"
    )

    num_fake_transitions = 50000
    for _ in range(num_fake_transitions):
        state = np.random.randn(state_dim).astype(np.float32)
        next_state = np.random.randn(state_dim).astype(np.float32)

        action_idx = np.random.randint(0, len(possible_bets))

        reward = np.random.randn() * 2.0
        done = np.random.rand() < 0.05
        agent.store_transition(state, action_idx, reward, next_state, done)

    num_training_steps = 20000
    losses = []
    for step in range(num_training_steps):
        loss_val = agent.update()
        if loss_val is not None:
            losses.append(loss_val)

        if (step+1) % 2000 == 0:
            avg_loss = np.mean(losses[-1000:]) if len(losses) >= 1000 else np.mean(losses)
            print(f"Step {step+1}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")

    test_state = np.array([0.0, 5.0, 1.2, -1.5, 2.0], dtype=np.float32)
    chosen_bet = agent.choose_bet_size(test_state)
    print(f"Chosen bet for test_state {test_state} is {chosen_bet}")

    return agent, losses


trained_agent, training_losses = train_betting_agent()