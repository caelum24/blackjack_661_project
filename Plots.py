import matplotlib.pyplot as plt
import numpy as np

def plot_basic_metrics(bankroll_history, loss_history, reward_history):

    plt.figure()
    plt.plot(bankroll_history)
    plt.title("Bankroll Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll")
    plt.show()

    # 2) Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 3) Reward
    plt.figure()
    plt.plot(reward_history)
    plt.title("Reward Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Reward")
    plt.show()


def plot_epsilon_bonus_decay(epsilon_values, bonus_values):
    plt.figure()
    plt.plot(epsilon_values)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Epsilon")
    plt.show()

    plt.figure()
    plt.plot(bonus_values)
    plt.title("Bonus Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bonus")
    plt.show()


def plot_hyperparameter_results(hparam_results):

    labels = list(hparam_results.keys())
    values = list(hparam_results.values())

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Hyperparameter Search Results")
    plt.xlabel("Hyperparameter Setting")
    plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.show()


def plot_comparative_strategies(basic_strategy_results, learned_strategy_results, counting_strategy_results=None):

    plt.figure()
    plt.plot(basic_strategy_results, label="Basic Strategy")
    plt.plot(learned_strategy_results, label="Learned Strategy")
    
    if counting_strategy_results is not None:
        plt.plot(counting_strategy_results, label="Counting Strategy")
    
    plt.title("Comparative Strategies")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll or Reward")
    plt.legend()
    plt.show()


def plot_action_values(action_values):
    
    actions = list(action_values.keys())
    values = list(action_values.values())

    plt.figure()
    plt.bar(actions, values)
    plt.title("Relative Value of Each Action")
    plt.xlabel("Action")
    plt.ylabel("Estimated Value")
    plt.show()


def plot_basic_metrics(bankroll_history, loss_history, reward_history):
    """
    Plots basic metrics:
    - Bankroll over time
    - Loss over epochs
    - Reward over episodes
    """
    # 1) Bankroll
    plt.figure()
    plt.plot(bankroll_history)
    plt.title("Bankroll Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll")
    plt.show()

    # 2) Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 3) Reward
    plt.figure()
    plt.plot(reward_history)
    plt.title("Reward Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Reward")
    plt.show()


def plot_epsilon_bonus_decay(epsilon_values, bonus_values):
    """
    Plot how epsilon and bonus change over training steps or episodes.
    """
    # 1) Epsilon decay
    plt.figure()
    plt.plot(epsilon_values)
    plt.title("Epsilon Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Epsilon")
    plt.show()

    # 2) Bonus decay
    plt.figure()
    plt.plot(bonus_values)
    plt.title("Bonus Decay Over Time")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bonus")
    plt.show()


def plot_hyperparameter_results(hparam_results):
    """
    Plot hyperparameter search results (e.g. best reward vs. hyperparameters).
    hparam_results could be a dictionary or list of (param, performance).
    Example structure:
      hparam_results = {
        'lr=1e-3, hidden=64': 0.56,
        'lr=1e-4, hidden=128': 0.58,
        ...
      }
    """
    labels = list(hparam_results.keys())
    values = list(hparam_results.values())

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Hyperparameter Search Results")
    plt.xlabel("Hyperparameter Setting")
    plt.ylabel("Performance Metric")
    plt.tight_layout()
    plt.show()


def plot_comparative_strategies(basic_strategy_results, learned_strategy_results, counting_strategy_results=None):
    """
    Compare different strategies (basic, learned, counting).
    Each input might be a list of rewards or bankroll over time.
    """
    plt.figure()
    plt.plot(basic_strategy_results, label="Basic Strategy")
    plt.plot(learned_strategy_results, label="Learned Strategy")
    
    if counting_strategy_results is not None:
        plt.plot(counting_strategy_results, label="Counting Strategy")
    
    plt.title("Comparative Strategies")
    plt.xlabel("Hand / Episode")
    plt.ylabel("Bankroll or Reward")
    plt.legend()
    plt.show()


def plot_action_values(action_values):
    """
    Shows a histogram (or other chart) of the relative values
    of each action in a given state (or aggregated states).
    
    `action_values` might be a dict or list of Q-values or predicted values:
      {
        'Hit': 0.12,
        'Stand': -0.02,
        'Double': 0.08,
        ...
      }
    """
    actions = list(action_values.keys())
    values = list(action_values.values())

    plt.figure()
    plt.bar(actions, values)
    plt.title("Relative Value of Each Action")
    plt.xlabel("Action")
    plt.ylabel("Estimated Value")
    plt.show()
