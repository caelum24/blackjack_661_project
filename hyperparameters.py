import torch

class HyperParameters:
    DECK_COUNT = 6  # Number of decks
    MIN_BET = 1
    MAX_BET = 100
    INITIAL_BANKROLL = 1000
    GAMMA = 0.99  # Discount factor
    MEMORY_SIZE = 10000  # Reduced replay buffer size for faster learning
    BATCH_SIZE = 128  # Increased batch size for more stable training
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995  # Slower epsilon decay
    LEARNING_RATE = 0.001  # Increased learning rate for faster learning
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TARGET_UPDATE = 1000  # How often to update the target network
