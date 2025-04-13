from numpy import exp

class ExponentialDecayer:

    def __init__(self, num_steps, decay_strength, e_min = 0, e_max = 1):

        self.num_steps = num_steps

        # typically between 1-10
        self.decay_strength = decay_strength
        self.epsilon = e_max
        self.e_max = e_max
        self.e_min = e_min

        self.decay_rate = self.decay_strength/self.num_steps

        # exponential parameter
        self.step_count = 0
    
    def get_epsilon(self):
        return self.epsilon
    
    def decay_epsilon(self) -> None:
        self.step_count += 1
        self.epsilon = self.e_min + (self.e_max - self.e_min) * exp(-self.decay_rate * self.step_count)


if __name__ == "__main__":
    # dec = EpsilonDecayer(decay_type="rbed", e_decay=0.001, reward_threshold= -1, reward_target = 0, reward_increment=0.001, alpha = 0.001)
    num_steps = 20000
    strength = 7
    dec = ExponentialDecayer(num_steps, strength, e_max=1, e_min=0.5)
    i = 0
    epsilons = []
    # while dec.reward_threshold < dec.reward_target:
    for i in range(num_steps):
        # reward = random.randint(-1, 2)
        # dec.update_reward_ema(reward)
        dec.decay_epsilon()
        i += 1
        # if i % 1000  == 0:
        # print(dec.get_epsilon(), dec.reward_ema, dec.reward_threshold)
        epsilons.append(dec.get_epsilon())

    print(epsilons[-1])
    from matplotlib import pyplot as plt
    plt.plot(range(len(epsilons)), epsilons)
    plt.show()
    # print(i)