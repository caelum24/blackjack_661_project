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

def train_agent(episodes=10000, update_target_every=100, print_every=100, agent=DQNAgent):
    env = BlackjackEnv()
    original_state_size = 6
    card_info_size = 10
    action_size = 3
    agent = DQNAgent(state_size=original_state_size + card_info_size, action_size=action_size)

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):
        # Reset environment
        state, _, _ = env.reset()

        # Place a bet
        bet_size_factor = agent.bet(state)
        state, _, _ = env.place_bet(bet_size_factor)

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
            next_state, reward, done = env.step(action)

            # Remember the experience
            agent.remember(state, action, reward, next_state, done)

            # Learn from experiences
            if len(agent.memory) >= HyperParameters.BATCH_SIZE:
                loss = agent.learn()
                if loss:
                    episode_loss += loss
                    steps += 1

            state = next_state
            cumulative_reward += reward

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
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon:.2f}, Loss: {avg_loss:.4f}, Bankroll: {env.bankroll}, Reward: {cumulative_reward}")

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(bankroll_history)
    plt.title('Bankroll History')
    plt.xlabel('Episode')
    plt.ylabel('Bankroll')

    plt.subplot(1, 3, 2)
    # Use rolling average for smoother curve
    rolling_avg = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title('Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 3, 3)
    # Use rolling average for loss as well
    if loss_history:
        rolling_loss = np.convolve(loss_history, np.ones(100)/100, mode='valid')
        plt.yscale('log')
        plt.plot(rolling_loss)
        plt.title('Average Loss (100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('blackjack_training_results.png')
    plt.show()

    return agent, env

def evaluate_agent(agent, env, episodes=1000):
    """Evaluate the trained agent's performance"""
    total_reward = 0
    wins = 0
    losses = 0
    pushes = 0

    for e in range(episodes):
        state, _, _ = env.reset()

        # Place a bet
        bet_size_factor = agent.bet(state)
        state, _, _ = env.place_bet(bet_size_factor)

        episode_reward = 0
        done = False

        while not done:
            # Determine valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)

            # Take action (no random exploration during evaluation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(HyperParameters.DEVICE)
            with torch.no_grad():
                act_values = agent.policy_net(state_tensor).cpu().data.numpy()[0]

            # Filter out invalid actions
            masked_values = np.copy(act_values)
            for i in range(agent.action_size):
                if i not in valid_actions:
                    masked_values[i] = -np.inf

            action = np.argmax(masked_values)
            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

        total_reward += episode_reward

        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            pushes += 1

    print(f"Evaluation results over {episodes} episodes:")
    print(f"Average reward: {total_reward / episodes:.4f}")
    print(f"Win rate: {wins / episodes:.4f}")
    print(f"Loss rate: {losses / episodes:.4f}")
    print(f"Push rate: {pushes / episodes:.4f}")

    return total_reward / episodes