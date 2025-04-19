# q_state_train.py
import os, numpy as np, matplotlib.pyplot as plt
from QStateAgent         import QStateAgent
from split_environment   import BlackjackEnv
from SplitTracker        import SplitTracker
from exponential_decay   import ExponentialDecayer
from RewardBonus         import RewardBonus
from hyperparameters     import HyperParameters
from q_state_modeling    import model

def train_q_agent(agent: QStateAgent,
                  episodes: int = 200_000,
                  update_target_every: int = 200,
                  print_every: int = 10_000,
                  save_dir: str = "final_models_dc_2"):

    os.makedirs(save_dir, exist_ok=True)
    env = BlackjackEnv(count_type=agent.count_type)

    eps_mgr   = ExponentialDecayer(episodes,
                                   decay_strength=HyperParameters.EPSILON_DECAY_STRENGTH,
                                   e_max=HyperParameters.EPSILON_START,
                                   e_min=HyperParameters.EPSILON_MIN)
    bonus_mgr = RewardBonus(episodes,
                            decay_strength=HyperParameters.EPSILON_DECAY_STRENGTH + 4,
                            initial_bonuses=HyperParameters.BONUS_REWARDS)

    bankroll_hist, reward_hist, loss_hist = [], [], []

    for ep in range(1, episodes + 1):
        state, _, _ = env.reset()
        if state[0] == 21:      # natural blackjack – skip
            continue

        split_tracker = None
        terminal_buffer = []
        done, ep_loss, steps = 0, 0.0, 0

        while done != 2:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward += bonus_mgr.get_bonus(action)

            if action == 3:                 # split chosen
                split_tracker = split_tracker or SplitTracker(state)
                split_tracker.split(next_state)

            elif done != 0:                 # hand finished
                terminal_buffer.append((state, action, reward,
                                        next_state, np.zeros_like(next_state), True))
                if done == 1:
                    split_tracker.switch_hand(next_state)

            else:                           # regular step
                ep_loss += agent.learn(state, action, reward,
                                        next_state, np.zeros_like(next_state), done=False)

            state = next_state
            steps += 1

        # dealer finished – update terminal buffers
        hand_rewards = env.deliver_rewards()
        for i, (s, a, r, sn, split_sn, _) in enumerate(terminal_buffer):
            ep_loss += agent.learn(s, a, r + hand_rewards[i], sn, split_sn, done=True)

        if split_tracker:
            for split_s, (h1_s, h2_s) in split_tracker.get_split_next_hands():
                agent.learn(split_s, 3, bonus_mgr.get_bonus(3), h1_s, h2_s, done=False)

        bankroll_hist.append(env.bankroll)
        reward_hist.append(sum(hand_rewards))
        loss_hist.append(ep_loss / max(1, steps))

        eps_mgr.decay_epsilon()
        agent.update_epsilon(eps_mgr.get_epsilon())

        if ep % update_target_every == 0:
            agent.update_target_q_table()
        if ep % print_every == 0:
            print(f"[{ep}/{episodes}] ε={agent.epsilon:.3f} "
                  f"Δloss={ep_loss/steps:.4f} Δreward={sum(hand_rewards):+} "
                  f"bankroll={env.bankroll}")

    # quick plots …
    # (remove if running headless)
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.plot(bankroll_hist); plt.title("Bankroll")
    plt.subplot(1,3,2); plt.plot(np.convolve(reward_hist,np.ones(500)/500,'valid')); plt.title("Reward (roll‑500)")
    plt.subplot(1,3,3); plt.plot(loss_hist); plt.title("Per‑step loss")
    plt.tight_layout(); plt.show()

    # save json
    agent.save_model(os.path.join(save_dir,
                     f"{agent.count_type}_q_state_learn_model.json"))
    return agent, env


# --- run --------------------------------------------------------------------
if __name__ == "__main__":
    for ct in ["empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
        print(f"\n=== Training table for {ct} ===")
        ag = QStateAgent(count_type=ct)
        trained_ag, trained_env = train_q_agent(ag,
                                                episodes=500_000,
                                                update_target_every=500,
                                                print_every=50_000)
        evaluate_agent(trained_ag, trained_env, episodes=50_000)
        model(trained_ag)
