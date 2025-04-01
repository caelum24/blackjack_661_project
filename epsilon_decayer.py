from numpy import exp, random
from collections import deque
from matplotlib import pyplot as plt

class EpsilonDecayer():

    def __init__(self, decay_type = "lin", e_max = 1, e_min = 0, e_decay = 0.001, reward_target = 0, reward_increment = 1, reward_threshold = 0, alpha = 0.005, reward_ema_init = -1):

        # assert which decay type is being used
        self.decay_type = decay_type
        
        # general epsilon parameters
        self.epsilon = e_max
        self.e_max = e_max
        self.e_min = e_min
        self.decay_rate = e_decay

        # exponential parameter
        self.step_count = 0
        
        # rbed parameters
        self.reward_target = reward_target
        self.steps_to_take = reward_target
        self.reward_increment = reward_increment
        self.reward_threshold = reward_threshold
        # self.reward_average = deque(maxlen=deque_size)
        self.reward_ema = reward_ema_init
        self.alpha = alpha

        self.decay_manager = {
            "linear" : self.linear_decay_epsilon,
            "exponential" : self.exponential_decay_epsilon,
            "rbed" : self.rbed_decay_epsilon
        }

        try:
            assert self.decay_type in self.decay_manager.keys()
        except AssertionError:
            print("ERROR: decay type must be one of", list(self.decay_manager.keys()))
            exit()
    
    def get_epsilon(self):
        return self.epsilon
    
    def update_reward_ema(self, reward):
        self.reward_ema = self.reward_ema * (1-self.alpha) + reward * self.alpha
    
    def decay_epsilon(self):
        self.decay_manager[self.decay_type]()

    def linear_decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.decay_rate, self.e_min)

    def exponential_decay_epsilon(self):
        self.step_count += 1
        self.epsilon = self.e_min + (self.e_max - self.e_min) * exp(-self.decay_rate * self.step_count)

    def rbed_decay_epsilon(self):
        # print(self.epsilon > self.e_min and self.reward_ema >= self.reward_threshold, self.epsilon > self.e_min, self.reward_ema >= self.reward_threshold)
        print(self.epsilon, self.e_min, self.reward_ema, self.reward_threshold)
        if self.epsilon > self.e_min and self.reward_ema >= self.reward_threshold:
            # print("incrementing", self.reward_threshold)
            self.epsilon -= self.decay_rate
            self.epsilon = max(self.epsilon, self.e_min)
            self.reward_threshold += self.reward_increment

if __name__ == "__main__":
    dec = EpsilonDecayer(decay_type="rbed", e_decay=0.001, reward_threshold= -1, reward_target = 0, reward_increment=0.001, alpha = 0.001)
    # print(dec.get_epsilon())
    # dec.decay_epsilon()
    # print(dec.get_epsilon())
    i = 0
    epsilons = []
    while dec.reward_threshold < dec.reward_target:
    # for i in range(10000):
        reward = random.randint(-1, 2)
        dec.update_reward_ema(reward)
        dec.decay_epsilon()
        i += 1
        # if i % 1000  == 0:
        print(dec.get_epsilon(), dec.reward_ema, dec.reward_threshold)
        epsilons.append(dec.get_epsilon())

    plt.plot(range(len(epsilons)), epsilons)
    plt.show()
    # print(i)