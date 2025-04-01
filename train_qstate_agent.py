
from QStateAgent import QStateAgent
from environment import BlackjackEnv
from matplotlib import pyplot as plt
import numpy as np


def train_q_agent(agent, env, episodes=100000, update_target_every=100, print_every=10000):

    # Statistics
    bankroll_history = []
    reward_history = []
    loss_history = []

    for e in range(episodes):
        state, reward, done = env.reset()

        episode_loss = 0
        steps = 0
        cumulative_reward = 0
        
        while not done:

            # Determine valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)
            
            action = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action)
            if str(reward) == "wtf":
                print("here")

            loss = agent.learn(state, action, reward, next_state, done)

            episode_loss += loss
            steps += 1
            state = next_state
            cumulative_reward += reward

        if e % update_target_every == 0:
            agent.update_target_q_table()

        # Store statistics
        bankroll_history.append(env.bankroll)
        reward_history.append(cumulative_reward)
        if steps > 0:
            loss_history.append(episode_loss / steps)
        else:
            loss_history.append(0)

        if e % print_every == 0:
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon:.2f}, Bankroll: {env.bankroll}, Reward: {cumulative_reward}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(bankroll_history)
    plt.title('Bankroll History')
    plt.xlabel('Episode')
    plt.ylabel('Bankroll')

    plt.subplot(1, 3, 2)
    rolling_avg = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title('Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 3, 3)
    rolling_avg = np.convolve(reward_history, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title('Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('q_learning_blackjack_training.png')
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
        # bet_size_factor = agent.bet(state)
        # state, _, _ = env.place_bet(bet_size_factor)

        episode_reward = 0
        done = False

        while not done:
            # Determine valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)

            # Take action (no random exploration during evaluation)
            action = agent.act(state, valid_actions=valid_actions, e_greedy = False)

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


if __name__ == "__main__":
    agent = QStateAgent(state_size = 5, action_size = 3)
    env = BlackjackEnv(num_decks = 6, count_type = "hi_lo")

    train_q_agent(agent, env, episodes=100000, update_target_every=100, print_every=1000)
    evaluate_agent(agent, env, episodes=1000)