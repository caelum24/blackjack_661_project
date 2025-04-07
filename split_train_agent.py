import numpy as np
import random
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
"""
Train agent and evaluate agent code. Import and use directly to print information and
get the accuracy returned. Hyperparameters and gathered from the hyperparameters file. 

Train agent is set up to run with the DQNAgent for now, might need to make modifications to work
with other models. 
"""

# TODO -> want to add input reward system for the trainer that is a hyperparameter

# TODO -> this function is the training loop for a model to play blackjack with
def train_agent(episodes=10000, update_target_every=100, print_every=100, agent:DQNAgent=DQNAgent(count_type="full")):
    
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
        # done in [0,1,2] -> 0 = not done, 1 = hand done (for splitting), 2 = completely done
        done = 0

        # TODO -> probably need to store the splits in  different environment to do it properly
        while done != 2:
            # TODO -> need to add some stuff about splitting
            # Take action
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            if action == 3:
                # if splitting, use split tracker to monitor the state of those hands
                if split_tracker is None:
                    # create split tracker to keep track of the splitting tree
                    split_tracker = SplitTracker(state)

                # add to the split tree the lower state that results
                split_tracker.split(next_state)
            
            else:
                # if not splitting, remember the experience for training
                agent.remember(state, action, reward, next_state, done)

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
            #NOTE -> the cumulative reward is going to be skewed based on the bonus system we implement
            cumulative_reward += reward
        
        # TODO TODO if agent split during the last game... look at the split tracker get states stuff

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
        #TODO -> if our environment returns 1, we need to store that for later so that we can go back and find the reward for each one
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
    