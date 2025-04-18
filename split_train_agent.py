import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from split_environment import BlackjackEnv 
from SplitTracker import SplitTracker
from DQNAgent import DQNAgent
from hyperparameters import HyperParameters
from datetime import datetime
from exponential_decay import ExponentialDecayer
from RewardBonus import RewardBonus
from QStateAgent import QStateAgent

"""
Train agent and evaluate agent code. Import and use directly to print information and
get the accuracy returned. Hyperparameters and gathered from the hyperparameters file. 

Train agent is set up to run with the DQNAgent for now, might need to make modifications to work
with other models. 
"""

def train_agent(agent, episodes=10000, update_target_every=100, print_every=100):
    
    # environment needs to embody the same counts as the agent
    env = BlackjackEnv(count_type=agent.count_type)
    epsilon_manager = ExponentialDecayer(episodes, decay_strength=HyperParameters.EPSILON_DECAY_STRENGTH, e_max=HyperParameters.EPSILON_START, e_min=HyperParameters.EPSILON_MIN)

    ### Bonus Reward Stuff ###
    bonus_manager = RewardBonus(episodes, decay_strength=HyperParameters.EPSILON_DECAY_STRENGTH+4, initial_bonuses=HyperParameters.BONUS_REWARDS)

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):

        # Reset environment
        state, reward, done = env.reset()

        # if blackjack, can't do anything, so just skip to next episode
        if state[0] == 21:
            continue

        cumulative_reward = 0
        episode_loss = 0
        steps = 0
        terminal_hand_states = []
        split_tracker:SplitTracker = None
        # done in [0,1,2] -> 0 = not done, 1 = hand done (for splitting), 2 = completely done
        done = 0

        while done != 2:
            # Take action
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # the bonus rewards added here
            reward += bonus_manager.get_bonus(action)

            if action == 3:
                # if splitting, use split tracker to monitor the state of those hands
                if split_tracker is None:
                    # create split tracker to keep track of the splitting tree
                    split_tracker = SplitTracker(state)

                # add to the split tree the lower state that results
                split_tracker.split(next_state)
            
            elif done != 0:
                # if done occurred in some way or another, we need to wait until the end of the game to determine what the reward was
                # store these hands in an array and come back to them later
                terminal_hand_states.append((state, action, reward, next_state, np.zeros_like(next_state), True))

                if done==1:
                    # if just a hand is done... only happens after a split
                    split_tracker.switch_hand(next_state)

            else:
                # if not done, we can immediately remember the experience
                agent.remember(state, action, reward, next_state, np.zeros_like(next_state), done=False)

            
            
            # Learn from experiences
            if len(agent.memory) >= HyperParameters.BATCH_SIZE:
                loss = agent.learn()
                if loss:
                    episode_loss += loss
                    steps += 1
            else:
                steps+=1

            # switch state to the next step
            state = next_state

        
        # add up the rewards of playing the game
        #NOTE -> the cumulative reward is going to be skewed based on the bonus system we implement
        hand_rewards = env.deliver_rewards()

        # currently, cumulative reward only cares about the game, not the bonuses
        cumulative_reward = sum(hand_rewards)

        # learn from the terminal states:
        for i, (state, action, reward, next_state, split_state, done) in enumerate(terminal_hand_states):
            # remember but also add the reward from the end of the game to actually get ahold of the states
            agent.remember(state, action, reward+hand_rewards[i], next_state, split_state, done) # done = True

        # if agent split during the game, store the states that resulted for split training
        if split_tracker is not None:
            split_next_hands = split_tracker.get_split_next_hands()
            for split_state, (ret1_state, ret2_state) in split_next_hands:
                action = 3 # 3 for splitting
                reward = bonus_manager.get_bonus(action)
                agent.remember(split_state, action, reward, ret1_state, ret2_state, done=False)
                # print("SPLITTING", split_state, action, reward, ret1_state, ret2_state)

        
        # Update target network
        if e % update_target_every == 0:
            agent.update_target_network()
        # Save model
        if e % (episodes // 10) == 0:
            save_model(agent, env, "checkpoint_finished_model" + str(e))
        # Store statistics
        bankroll_history.append(env.bankroll)
        reward_history.append(cumulative_reward)
        if steps > 0:
            loss_history.append(episode_loss / steps)
        else:
            loss_history.append(0)

        # update epsilon
        epsilon_manager.decay_epsilon()
        agent.update_epsilon(epsilon_manager.get_epsilon())
    
        # decay the bonuses
        bonus_manager.decay_bonuses()

        # Print progress
        if e % print_every == 0:
            avg_loss = np.mean(loss_history[-print_every:]) if loss_history else 0
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon:.2f}, Loss: {avg_loss:.4f}, Bankroll: {env.bankroll}, Reward: {cumulative_reward}")

    return agent, env

def evaluate_agent(agent, env:BlackjackEnv, episodes=1000):
    """Evaluate the trained agent's performance"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_history = []
    total_reward = 0
    # wins = 0
    # losses = 0
    # pushes = 0
    for e in range(episodes):
        done = 0
        state, _, done = env.reset()

        # Place a bet
        # bet_size_factor = agent.bet(state)
        # state, _, _ = env.place_bet(bet_size_factor)

        while done != 2:

            action = agent.act(state)
            next_state, _, done = env.step(action)

            state = next_state

        episode_reward = sum(env.deliver_rewards())
        total_reward += episode_reward
        reward_history.append(episode_reward)

    print(f"Evaluation results over {episodes} episodes:")
    print(f"Average reward: {total_reward / episodes:.4f}")
    # print(f"Win rate: {wins / episodes:.4f}")
    # print(f"Loss rate: {losses / episodes:.4f}")
    # print(f"Push rate: {pushes / episodes:.4f}")

    # saving graphs stuff
    graph_path = os.path.join('graphs', f'evaluation_results_{timestamp}.png')
    plt.savefig(graph_path)
    plt.close()
    print(f"Evaluation graphs saved to {graph_path}")

    return total_reward / episodes


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

if __name__ == "__main__":

    # agent = DQNAgent(count_type="hi_lo")
    for count_type in ["empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
        agent = QStateAgent(count_type=count_type)
        agent, env = train_agent(episodes=1000000, update_target_every=100, print_every=20000, agent=agent)
        evaluate_agent(agent, env, episodes=100000)
        agent.save_model(f"{count_type}_q_state_learn_model.json")
