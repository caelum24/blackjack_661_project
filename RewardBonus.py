from exponential_decay import ExponentialDecayer

class RewardBonus:

    def __init__(self, episodes, decay_strength, initial_bonuses):

        self.hit_bonus_manager = ExponentialDecayer(episodes, decay_strength, e_max = initial_bonuses[0])
        self.stand_bonus_manager = ExponentialDecayer(episodes, decay_strength, e_max = initial_bonuses[1])
        self.double_bonus_manager = ExponentialDecayer(episodes, decay_strength, e_max = initial_bonuses[2])
        self.split_bonus_manager = ExponentialDecayer(episodes, decay_strength, e_max = initial_bonuses[3])

    def decay_bonuses(self):

        self.hit_bonus_manager.decay_epsilon()
        self.stand_bonus_manager.decay_epsilon()
        self.double_bonus_manager.decay_epsilon()
        self.split_bonus_manager.decay_epsilon()

    def get_bonus(self, action):
        if action == 0:
            return self.hit_bonus
        elif action == 1:
            return self.stand_bonus
        elif action == 2:
            return self.double_bonus
        elif action == 3:
            return self.split_bonus
        

    @property
    def hit_bonus(self):
        return self.hit_bonus_manager.get_epsilon()

    @property
    def stand_bonus(self):
        return self.stand_bonus_manager.get_epsilon()

    @property
    def double_bonus(self):
        return self.double_bonus_manager.get_epsilon()
    
    @property
    def split_bonus(self):
        return self.split_bonus_manager.get_epsilon()

