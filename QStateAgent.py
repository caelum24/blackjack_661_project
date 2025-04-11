import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from environment import BlackjackEnv 
from hyperparameters import HyperParameters
from collections import defaultdict
from epsilon_decayer import EpsilonDecayer

# TODO -> implement the Epsilon Decayer


class QStateAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = HyperParameters.LEARNING_RATE
        self.gamma = HyperParameters.GAMMA
        self.epsilon = HyperParameters.EPSILON_START
        self.epsilon_min = HyperParameters.EPSILON_MIN
        self.epsilon_decay = HyperParameters.EPSILON_DECAY

        self.q_table = defaultdict(lambda: np.zeros(action_size))  # Training Q-values
        self.target_q_table = defaultdict(lambda: np.zeros(action_size))  # Target Q-values

        self.iter_num = 0

    def update_target_q_table(self):
        for key in self.q_table.keys():
            self.target_q_table[key] = np.copy(self.q_table[key])

    # def bet() TODO -> add ability to bet?

    def act(self, state, valid_actions=None, e_greedy = True):
        '''
            This function takes in a state and valid actions and outputs an action for the agent 
            to take based on e-greedy policy exploration
        '''
        state=tuple(state)

        # choose an action based on a state
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        # decide if to explore with a random action
        if np.random.rand() <= self.epsilon and e_greedy:
            # take a random action
            return random.choice(valid_actions)


        masked_values = np.copy(self.target_q_table[state])
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_values[i] = -np.inf

        # get most rewarded action
        return np.argmax(masked_values)

       
    def learn(self, state, action, reward, next_state, done):

        state = tuple(state)
        next_state = tuple(next_state)

        # get the q value of the state-action taken
        q_value = self.q_table[state][action]

        # use the target table to get the next value
        next_q_value = self.target_q_table[next_state]

        # get the expected total reward from this action and future actions
        expected_q_value = reward + (1 - done) * self.gamma * np.max(next_q_value)

        loss = expected_q_value - q_value

        # update the q value based on the loss
        new_q_value = q_value + self.lr * loss

        # update the q_table to reflect the new value
        self.q_table[state][action] = new_q_value

        self.epsilon = max(self.epsilon - 0.000001, self.epsilon_min)

        return loss

