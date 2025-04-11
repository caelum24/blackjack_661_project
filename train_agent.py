import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from environment import BlackjackEnv 
from DQNAgent import DQNAgent
from hyperparameters import HyperParameters
import os
from datetime import datetime
"""
Train agent and evaluate agent code. Import and use directly to print information and
get the accuracy returned. Hyperparameters and gathered from the hyperparameters file. 

Train agent is set up to run with the DQNAgent for now, might need to make modifications to work
with other models. 
"""
def save_model(agent, env, timestamp):
    """Save the trained model and environment state"""
    model_path = os.path.join('models', f'blackjack_agent_{timestamp}.pth')
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'env_bankroll': env.bankroll
    }, model_path)
    print(f"Model saved to {model_path}")

def save_training_graphs(bankroll_history, reward_history, loss_history, timestamp):
    """Save training graphs with improved titles and formatting"""
    plt.figure(figsize=(15, 5))

    # Bankroll History
    plt.subplot(1, 3, 1)
    plt.plot(bankroll_history)
    plt.title('Bankroll History Over Training')
    plt.xlabel('Episode')
    plt.ylabel('Bankroll')
    plt.grid(True)

    # Reward History
    plt.subplot(1, 3, 2)
    rolling_avg = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title('Average Reward (100-episode Rolling Average)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # Loss History
    plt.subplot(1, 3, 3)
    if loss_history:
        rolling_loss = np.convolve(loss_history, np.ones(100)/100, mode='valid')
        plt.yscale('log')
        plt.plot(rolling_loss)
        plt.title('Average Loss (100-episode Rolling Average)')
        plt.xlabel('Episode')
        plt.ylabel('Loss (log scale)')
        plt.grid(True)

    plt.tight_layout()
    graph_path = os.path.join('graphs', f'training_results_{timestamp}.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Training graphs saved to {graph_path}")

def train_agent(episodes=10000, update_target_every=1000, print_every=100, agent=DQNAgent):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = BlackjackEnv(count_type="empty")  # Using simplified state for now
    state_size = 4  # player_sum, dealer_up_card, usable_ace, can_double
    action_size = 3  # hit, stand, double
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):
        # Reset environment
        state, _, _ = env.reset()

        # Place a fixed bet for now (will be replaced by betting network)
        state, _, _ = env.place_bet(0.5)  # Fixed bet size factor

        cumulative_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            # Determine valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)

            # Take action
            action = agent.act(state, valid_actions)
            next_state, strategy_reward, done = env.step(action)

            # Remember the experience
            agent.remember(state, action, strategy_reward, next_state, done)

            # Learn from experiences
            if len(agent.memory) >= HyperParameters.BATCH_SIZE:
                loss = agent.learn()
                if loss:
                    episode_loss += loss
                    steps += 1

            state = next_state
            cumulative_reward += strategy_reward

        # Update target network
        if e % update_target_every == 0:
            agent.update_target_network()

        # Store statistics
        bankroll_history.append(env.bankroll)
        reward_history.append(cumulative_reward)
        if steps > 0:
            loss_history.append(episode_loss / steps)
        else:
            loss_history.append(0)

        # Print progress
        if e % print_every == 0:
            avg_loss = np.mean(loss_history[-print_every:]) if loss_history else 0
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon:.2f}, Loss: {avg_loss:.4f}, Bankroll: {env.bankroll}, Strategy Reward: {cumulative_reward}")

    # Save model and graphs
    save_model(agent, env, timestamp)
    save_training_graphs(bankroll_history, reward_history, loss_history, timestamp)

    return agent, env

def evaluate_agent(agent, env, episodes=1000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_reward = 0
    reward_history = []
    
    for episode in range(episodes):
        state, _, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            valid_actions = [0, 1]
            if state[3] == 1:
                valid_actions.append(2)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(HyperParameters.DEVICE)
            with torch.no_grad():
                act_values = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

            masked_values = np.copy(act_values)
            for i in range(agent.action_size):
                if i not in valid_actions:
                    masked_values[i] = -np.inf

            action = np.argmax(masked_values)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        reward_history.append(episode_reward)

    # Plot and save evaluation results
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title(f'Evaluation Results ({episodes} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    graph_path = os.path.join('graphs', f'evaluation_results_{timestamp}.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Evaluation graphs saved to {graph_path}")

    return total_reward / episodes