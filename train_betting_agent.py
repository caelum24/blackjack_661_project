import torch
import numpy as np
from environment import BlackjackEnv
from DQNAgent import DQNAgent
from main import load_model
from BettingAgent import BettingRLAgent
from replay_buffer import ReplayBuffer
from environment import BlackjackEnv

def load_and_run_model(model_path="dueling_dqn.pth", num_episodes=1000):
    """
    Loads a saved DQN model, then runs multiple blackjack episodes.
    Logs each episode's relevant info: card counts, final outcome, etc.
    """

    agent, env = load_model(model_path)


    all_results = []

    for episode in range(num_episodes):
        # Reset environment
        state, _, done = env.reset()

        episode_reward = 0
        step_count = 0

        start_hi_lo = env.hi_lo_count
        start_zen = env.zen_count
        start_uston = env.uston_apc_count

        while not done:
            # Decide valid actions
            valid_actions = [0, 1]  # hit, stand
            if state[3] == 1:       # can double
                valid_actions.append(2)

            # Agent picks action (no random exploration)
            action = agent.act(state, valid_actions=valid_actions)

            # Environment executes action
            next_state, reward, done = env.step(action)

            # Accumulate reward
            episode_reward += reward
            step_count += 1
            state = next_state

        episode_data = {
            "episode_index": episode,
            "steps": step_count,
            "final_reward": episode_reward,
            "hi_lo_count_end": env.hi_lo_count,
            "zen_count_end": env.zen_count,
            "uston_count_end": env.uston_apc_count,
            "hi_lo_count_start": start_hi_lo,
            "zen_count_start": start_zen,
            "uston_count_start": start_uston,
            "final_bankroll": env.bankroll,
        }
        all_results.append(episode_data)

    return all_results

def extract_betting_state(data_point):
    """
    Convert a game data point into a betting state.
    You can customize this — here we use just hi_lo count and bankroll.
    """
    hi_lo = data_point["hi_lo_count_start"]
    bankroll = data_point["final_bankroll"] / 1000  #Normalize bankroll
    return np.array([hi_lo, bankroll], dtype=np.float32)

def train_betting_agent_from_run_data(run_data, possible_bets, state_dim=2, num_training_steps=5000):
    agent = BettingRLAgent(
        state_dim=state_dim,
        possible_bets=possible_bets,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1e-5,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=200,
        device="cpu"
    )

    # Fill replay buffer with transitions
    for data in run_data:
        state = extract_betting_state(data)
        next_state = state
        done = True

        for i, bet in enumerate(possible_bets):
            reward = data["final_reward"] * bet
            agent.store_transition(state, i, reward, next_state, done)

    losses = []
    for step in range(num_training_steps):
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        if (step + 1) % 500 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)
            print(f"Step {step+1}, Avg Loss (last 100): {avg_loss:.4f}, Epsilon: {agent.epsilon:.2f}")

    return agent, losses

def evaluate_agent(betting_agent, playing_agent, env, episodes=1000):
    """Evaluate the trained agent's performance"""
    total_reward = 0
    wins = 0
    losses = 0
    pushes = 0

    for e in range(episodes):
        state, _, _ = env.reset()

        betting_state = np.array([
            env.hi_lo_count,
            env.bankroll / 1000.0,
        ], dtype=np.float32)

        bet_amount = betting_agent.act(betting_state)

        episode_reward = 0
        done = False

        while not done:
            # Determine valid actions
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)

            # Take action (no random exploration during evaluation)
            action = playing_agent.act(state, valid_actions=valid_actions)

            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

        total_reward += episode_reward *bet_amount

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
