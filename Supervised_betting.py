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