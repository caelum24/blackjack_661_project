from torch import cuda, device

class HyperParameters:
    DECK_COUNT = 2  # Number of decks
    MIN_BET = 1
    MAX_BET = 100
    INITIAL_BANKROLL = 1000
    GAMMA = 0.9  # Discount factor
    MEMORY_SIZE = 1000  # Reduced replay buffer size for faster learning
    BATCH_SIZE = 64
    EPSILON_START = 1.0 # for a training circumstance im doing, change this if you are starting over
    EPSILON_MIN = 0.001
    EPSILON_DECAY_STRENGTH = 4
    EPSILON_DECAY = 0.98  # Slower epsilon decay
    LEARNING_RATE = 0.0001  # Increased learning rate for faster learning
    DEVICE = device('cuda' if cuda.is_available() else 'cpu')
    TARGET_UPDATE = 300  # How often to update the target network

    # hit, stand, double, split bonus rewards that decay over time
    BONUS_REWARDS = [0.0, 0.0, 0.0, 0.0]