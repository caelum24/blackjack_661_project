from collections import deque
import random
import numpy as np
"""
Replay buffer to store past states. Import and use the capacity to control how many past states
are saved. 
"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, split_state, done):
        self.buffer.append((state, action, reward, next_state, split_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, split_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, split_state, done

    def __len__(self):
        return len(self.buffer)