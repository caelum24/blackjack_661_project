import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from environment import BlackjackEnv 
from hyperparameters import HyperParameters
from collections import defaultdict
from epsilon_decayer import EpsilonDecayer
import json

# TODO -> implement the Epsilon Decayer


class QStateAgent:

    counting_type_state_size = {"empty":5, "hi_lo":6, "zen":6, "uston_apc":6, "ten_count":6}

    def __init__(self, count_type):
        #NOTE -> this does not include full because the dictionary would be too sparse (curse of dimensionality on this one)
        if count_type not in ["empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
            print("count type must be one of", ["empty", "hi_lo", "zen", "uston_apc", "ten_count"])
            raise ValueError 

        # create empty memory array to fit with the api for split_train_agent
        self.memory = []

        self.count_type = count_type
        self.state_size = self.counting_type_state_size[count_type]
        self.action_size = 4 # hit, stand, double, split

        self.lr = HyperParameters.LEARNING_RATE
        self.gamma = HyperParameters.GAMMA
        self.epsilon = HyperParameters.EPSILON_START
        # self.epsilon_min = HyperParameters.EPSILON_MIN
        # self.epsilon_decay = HyperParameters.EPSILON_DECAY -> deprecated, as epsilon is dealt with during training and updated here with a method call

        # creating the "Models" for q-learning
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))  # Training Q-values
        self.target_q_table = defaultdict(lambda: np.zeros(self.action_size))  # Target Q-values

    def update_target_network(self):
        # implemented to make the apis work out for the split_train_agent class
        self.update_target_q_table()

    def update_target_q_table(self):
        for key in self.q_table.keys():
            self.target_q_table[key] = np.copy(self.q_table[key])

    def update_epsilon(self, epsilon): 
        # takes in the new epsilon from training loop to make updates properly with e-greedy policy
        self.epsilon = epsilon

    def act(self, state, eval=False):
        '''
            This function takes in a state and valid actions and outputs an action for the agent 
            to take based on e-greedy policy exploration
        '''
        state=tuple(state)

        # Determine valid actions to ensure model doesn't do something illegal
        valid_actions = [0, 1]  # Hit and stand are always valid
        if state[3] == 1:  # Can double
            valid_actions.append(2)
        if state[4] == 1: # can split
            valid_actions.append(3)
        
        # decide if to explore with a random action
        if not eval and np.random.rand() <= self.epsilon:
            # take a random action
            return random.choice(valid_actions)

        masked_values = np.copy(self.target_q_table[state])
        for i in range(self.action_size):
            if i not in valid_actions:
                masked_values[i] = -np.inf

        # get most rewarded action
        return np.argmax(masked_values)

    def remember(self, state, action, reward, next_state, split_next_state, done):
        self.learn(state, action, reward, next_state, split_next_state, done)
       
    def learn(self, state, action, reward, next_state, split_next_state, done):

        # decided not to implement batching for the q learning, instead, we update as they come in

        state = tuple(state)
        next_state = tuple(next_state)
        split_next_state = tuple(split_next_state)

        # determine if you had a split that you need to learn from the other state
        second_state_used = False
        for val in split_next_state:
            if val != 0:
                second_state_used = True
                done = False

        # get the q value of the state-action taken
        q_value = self.q_table[state][action]


        next_q_value1 = np.max(self.target_q_table[next_state])

        if second_state_used:
            next_q_value2 = np.max(self.target_q_table[split_next_state])
        else:
            next_q_value2 = 0

        combined_next_q_values = next_q_value1 + next_q_value2

        expected_q_value = reward + (1-done) * self.gamma * combined_next_q_values

        loss = expected_q_value - q_value

        # update the q value based on the loss
        new_q_value = q_value + self.lr * loss

        # update the q_table to reflect the new value
        self.q_table[state][action] = new_q_value

        return loss

    
    def load_model(self, q_table_file):
        with open(q_table_file, 'r') as f:
            loaded_q_table = json.load(f)

        self.q_table = loaded_q_table

    def save_model(self, filename = None):
        if filename is not None:
            filename = filename
        else:
            filename = "qtrain_model_checkpoint"

        # save the dictionary into the file
        with open(filename, "w") as f:
            json.dump(self.target_q_table, f)


