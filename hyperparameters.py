
class HyperParameters:
    DECK_COUNT = 6  # Number of decks
    MIN_BET = 1
    MAX_BET = 100
    INITIAL_BANKROLL = 1000
    GAMMA = 0.95  # Discount factor
    MEMORY_SIZE = 10000  # Replay buffer size
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_RATE = 0.001