import os
from matplotlib import pyplot as plt
from DQNAgent import DQNAgent
from split_environment import BlackjackEnv
from BasicStrategyAgent import BasicStrategyAgent
from QStateAgent import QStateAgent

def evaluate_agent(agent, env, episodes=10000):
    """Evaluate the trained agent's performance"""
    reward_history = []
    total_reward_history = []
    total_reward = 0
    wins = 0
    losses = 0
    pushes = 0
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

        episode_rewards = env.deliver_rewards()
        for reward in episode_rewards:
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                pushes += 1
        total_episode_reward = sum(episode_rewards)
        total_reward += total_episode_reward
        total_reward_history.append(total_reward)
        reward_history.append(total_episode_reward)

    average_reward = total_reward / episodes
    win_rate = wins / episodes
    loss_rate = losses / episodes
    push_rate = pushes / episodes
    print(f"Evaluation results over {episodes} episodes:")
    print(f"Average reward: {average_reward:.4f}")
    # print(f"Win rate: {wins / episodes:.4f}")
    # print(f"Loss rate: {losses / episodes:.4f}")
    # print(f"Push rate: {pushes / episodes:.4f}")

    return total_reward_history, average_reward, win_rate, loss_rate, push_rate

def compare_agent_models(input_folder, output_path, include_basic = True):

    for count_type in ['empty', 'hi_lo', 'zen', 'uston_apc', 'ten_count',  'full']:
        weight_file = f"{input_folder}/blackjack_agent_{count_type}_FINAL.pth"
        # weight_file = f"{input_folder}/q_state_blackjack_agent_{count_type}_FINAL.json"
        agent:DQNAgent=DQNAgent(count_type=count_type)
        # agent = QStateAgent(count_type=count_type)
        agent.update_epsilon(0.0)
        agent.load_model(weight_file)
        env = BlackjackEnv(num_decks=6, count_type=count_type)
        total_reward_history, average_reward, win_rate, loss_rate, push_rate = evaluate_agent(agent, env, episodes=100000)
        plt.plot(range(len(total_reward_history)), total_reward_history, label = count_type)
        print(count_type, "Ave Reward:", average_reward, "WR:", win_rate, "LR:", loss_rate, "PR:", push_rate)
    
    if include_basic:
        # play with basic strategy
        agent = BasicStrategyAgent()
        env = BlackjackEnv(num_decks=6, count_type="empty")
        total_reward_history, average_reward, win_rate, loss_rate, push_rate = evaluate_agent(agent, env, episodes=100000)
        plt.plot(range(len(total_reward_history)), total_reward_history, label = "Basic Strategy")
        print(count_type, "Ave Reward:", average_reward, "WR:", win_rate, "LR:", loss_rate, "PR:", push_rate)

    plt.title("Comparative Analysis of Blackjack For Card Count Methods")
    plt.ylabel("Cumulative Reward")
    plt.xlabel("Episodes")
    plt.legend()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    input_folder = "final_models_dc_6"
    output_path = "final_models_dc_6/comp_plot.png"
    compare_agent_models(input_folder, output_path, include_basic=True)