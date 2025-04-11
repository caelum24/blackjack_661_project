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
"""
Train agent and evaluate agent code. Import and use directly to print information and
get the accuracy returned. Hyperparameters and gathered from the hyperparameters file. 

Train agent is set up to run with the DQNAgent for now, might need to make modifications to work
with other models. 
"""

# TODO -> want to add input reward system for the trainer that is a hyperparameter

# TODO -> this function is the training loop for a model to play blackjack with
def train_agent(agent, episodes=10000, update_target_every=100, print_every=100):
    
    # environment needs to embody the same counts as the agent
    env = BlackjackEnv(count_type=agent.count_type)

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):

        # Place a bet
        # TODO _> this part should go in a separate model training loop
        # counts = env.get_count() # returns a np array
        # counts = torch.FloatTensor(counts).to(agent.DEVICE)
        # bet_size_factor = betting_agent.bet(counts)
        # state, _, _ = env.place_bet(bet_size_factor)
        # player_bet = 1
        # Reset environment
        # TODO -> make it so reset takes in a bet for betting training loop in a new file
        state, reward, done = env.reset()

        cumulative_reward = 0
        episode_loss = 0
        steps = 0
        split_tracker:SplitTracker = None
        # done in [0,1,2] -> 0 = not done, 1 = hand done (for splitting), 2 = completely done
        done = 0

        while done != 2:
            # TODO -> need to add some stuff about splitting
            # Take action
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            # TODO -> add in the bonus rewards here

            if action == 3:
                # if splitting, use split tracker to monitor the state of those hands
                if split_tracker is None:
                    # create split tracker to keep track of the splitting tree
                    split_tracker = SplitTracker(state)

                # add to the split tree the lower state that results
                split_tracker.split(next_state)
            
            else:
                # if not splitting, remember the experience for training
                agent.remember(state, action, reward, next_state, np.zeros_like(next_state), (done==1 or done==2))

            if done == 1:
                # if just a hand is done... only happens after a split
                split_tracker.switch_hand(next_state)
            
            # Learn from experiences
            if len(agent.memory) >= HyperParameters.BATCH_SIZE:
                loss = agent.learn()
                if loss:
                    episode_loss += loss
                    steps += 1

            # switch state to the next step
            state = next_state

        
        # add up the rewards of playing the game
        #NOTE -> the cumulative reward is going to be skewed based on the bonus system we implement
        cumulative_reward = sum(env.deliver_rewards())
        
        # if agent split during the game, store the states that resulted for split training
        if split_tracker is not None:
            split_next_hands = split_tracker.get_split_next_hands()
            for split_state, (ret1_state, ret2_state) in split_next_hands:
                #TODO -> add in the split bonus rewards here
                reward = 0
                action = 3 # 3 for splitting
                agent.remember(split_state, action, reward, ret1_state, ret2_state, done=False)


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

    return agent, env

def evaluate_agent(agent, env:BlackjackEnv, episodes=1000):
    """Evaluate the trained agent's performance"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_history = []
    total_reward = 0
    # wins = 0
    # losses = 0
    # pushes = 0
    #TODO -> need to update this to work with the new splitting agent and environment
    for e in range(episodes):
        done = 0
        state, _, done = env.reset()

        # Place a bet
        # bet_size_factor = agent.bet(state)
        # state, _, _ = env.place_bet(bet_size_factor)

        while done != 2:

            action = agent.act(state)
            next_state, reward, done = env.step(action)

            state = next_state

        episode_reward = sum(env.deliver_rewards())
        total_reward += episode_reward
        reward_history.append(episode_reward)

    # print(f"Evaluation results over {episodes} episodes:")
    # print(f"Average reward: {total_reward / episodes:.4f}")
    # print(f"Win rate: {wins / episodes:.4f}")
    # print(f"Loss rate: {losses / episodes:.4f}")
    # print(f"Push rate: {pushes / episodes:.4f}")

    # saving graphs stuff
    # graph_path = os.path.join('graphs', f'evaluation_results_{timestamp}.png')
    # plt.savefig(graph_path)
    # plt.close()
    # print(f"Evaluation graphs saved to {graph_path}")

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

    agent = DQNAgent(count_type="hi_lo")
    train_agent(episodes=4, update_target_every=2, print_every=2, agent=agent)
