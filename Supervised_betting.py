import torch
import numpy as np
from environment import BlackjackEnv
from DQNAgent import DQNAgent
from main import load_model
from BettingAgent import BettingRLAgent
from replay_buffer import ReplayBuffer
from environment import BlackjackEnv

def collect_dictionary(agent, env, num_episodes):

    count_reward_dict = {}
    for e in range(num_episodes):
        done = 0
        state, _, done = env.reset()

        start_hi_lo = env.hi_lo_count
        start_zen = env.zen_count
        start_uston = env.uston_apc_count

        count_state = [start_hi_lo, start_zen, start_uston]
        count_state = tuple(np.round(count_state, 4))

        while done != 2:

            action = agent.act(state)
            next_state, _, done = env.step(action)

            state = next_state

        episode_reward = sum(env.deliver_rewards())

        if episode_reward != 0:
            label = 1 if episode_reward > 0 else 0

            if count_state in count_reward_dict:
                prev_avg, count = count_reward_dict[count_state]
                new_avg = (prev_avg * count + label) / (count + 1)
                count_reward_dict[count_state] = (new_avg, count + 1)
            else:
                count_reward_dict[count_state] = (label, 1)

        if (e + 1) % 100 == 0:
            print(f"Episode {e+1}/{num_episodes} — Last Reward: {episode_reward:.2f} — Count State: {count_state}")

    return count_reward_dict


def evaluate_nn_betting_agent(betting_model, playing_agent, env, episodes=1000, max_bet=100, min_bet=1):
    betting_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betting_model.to(device)

    total_reward = 0
    wins = 0
    losses = 0
    pushes = 0

    for e in range(episodes):
        state, _, _ = env.reset()

        # Get betting input from environment
        betting_state = np.array([
            env.hi_lo_count,
            env.zen_count,
            env.uston_apc_count
        ], dtype=np.float32)

        # Convert to tensor and get win probability
        with torch.no_grad():
            input_tensor = torch.tensor(betting_state).unsqueeze(0).to(device)  # shape [1, 3]
            win_prob = betting_model(input_tensor).item()  # scalar

        # Apply Kelly Criterion
        b = 1  # even money payout
        p = win_prob
        q = 1 - p
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(kelly_fraction, 0)  # only bet if edge
        bet_fraction = min(kelly_fraction, 1)
        bet_amount = max(min_bet, round(bet_fraction * max_bet))

        episode_reward = 0
        done = False

        while done != 2:
            action = playing_agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state

        hand_rewards = env.deliver_rewards()
        episode_reward = sum(hand_rewards)
        total_reward += episode_reward * bet_amount

        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            pushes += 1

    print(f"\nEvaluation over {episodes} episodes:")
    print(f"Average profit per hand: {total_reward / episodes:.4f}")
    print(f"Win rate: {wins / episodes:.4f}")
    print(f"Loss rate: {losses / episodes:.4f}")
    print(f"Push rate: {pushes / episodes:.4f}")
    return total_reward / episodes
