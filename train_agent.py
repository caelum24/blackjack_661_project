import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from split_environment import BlackjackEnv 
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



# TODO -> want to add input reward system for the trainer that is a hyperparameter

def train_agent(agent:DQNAgent, episodes=10000, update_target_every=100, print_every=100):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    deck_count = HyperParameters.DECK_COUNT
    env = BlackjackEnv(num_decks=deck_count, count_type=agent.count_type)

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):
        cumulative_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        # Reset environment
        state, _, done = env.reset()

        # Place a fixed bet for now (will be replaced by betting network)
        # state, _, _ = env.place_bet(0.5)  # Fixed bet size factor

        while not done:

            # Take action
            action = agent.act(state)
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
    # save_model(agent, env, timestamp)
    # save_training_graphs(bankroll_history, reward_history, loss_history, timestamp)

    return agent, env

def evaluate_agent(agent, env, episodes=1000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_reward = 0
    reward_history = []
    
    for episode in range(episodes):
        episode_reward = 0
        done = False
        state, _, done = env.reset()
        
        while not done:
            action = agent.act(state, eval=True)
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