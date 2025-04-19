
from QStateAgent import QStateAgent
from split_environment import BlackjackEnv
from matplotlib import pyplot as plt
import numpy as np
from q_state_modeling import model
from SplitTracker import SplitTracker
from hyperparameters import HyperParameters
from exponential_decay import ExponentialDecayer
from RewardBonus import RewardBonus

# q_state_train.py  ── dedicate loop for tabular Q‑learning
import os
import numpy as np
import matplotlib.pyplot as plt

from split_environment import BlackjackEnv
from SplitTracker        import SplitTracker
from exponential_decay   import ExponentialDecayer
from RewardBonus         import RewardBonus
from hyperparameters     import HyperParameters
from QStateAgent         import QStateAgent


def train_q_agent(agent: QStateAgent,
                  episodes: 200000,
                  update_target_every = 200,
                  print_every = 10000,
                  save_dir= "final_models_dc_2"):
    """
    Train QStateAgent.
    Splitting is handled through SplitTracker and the agents
    """

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
        if state[0] == 21:          # skip natural BJ
            continue

        split_tracker = None
        terminal_buffer = []        # (s, a, r, s', split_s', done)

        done, ep_loss, steps = 0, 0.0, 0

        while done != 2:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            reward += bonus_mgr.get_bonus(action)    # optional shaping

            if action == 3:                          # chose to split
                if split_tracker is None:
                    split_tracker = SplitTracker(state)
                split_tracker.split(next_state)

            elif done != 0:   

                terminal_buffer.append( (state, action, reward,
                                          next_state, np.zeros_like(next_state), True) )
                if done == 1:
                    split_tracker.switch_hand(next_state)
            else:
                step_loss = agent.learn(state, action, reward,
                                        next_state, np.zeros_like(next_state), done=False)
                ep_loss += step_loss

            state = next_state
            steps += 1

        # final dealer result
        hand_rewards = env.deliver_rewards() 
        total_reward = sum(hand_rewards)

        # learn from buffered terminal states
        for i,(s,a,r,sn,split_n,_) in enumerate(terminal_buffer):
            final_r = r + hand_rewards[i]
            step_loss = agent.learn(s,a,final_r,sn,split_n,done=True)
            ep_loss += step_loss

        if split_tracker is not None:
            for split_s, (h1_s, h2_s) in split_tracker.get_split_next_hands():
                bonus_r = bonus_mgr.get_bonus(3) 
                agent.learn(split_s, 3, bonus_r,
                            h1_s, h2_s, done=False)

        bankroll_hist.append(env.bankroll)
        reward_hist.append(total_reward)
        loss_hist.append(ep_loss / max(steps,1))

        eps_mgr.decay_epsilon()
        agent.update_epsilon(eps_mgr.get_epsilon())

        if ep % update_target_every == 0:
            agent.update_target_q_table()

        if ep % print_every == 0:
            print(f"[{ep:>8}/{episodes}]  ε={agent.epsilon:.3f} "
                  f"Δloss={ep_loss/steps:.4f}  Δreward={total_reward:+.2f} "
                  f"bankroll={env.bankroll}")

    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1); plt.plot(bankroll_hist); plt.title("Bankroll")
    plt.subplot(1,3,2); plt.plot(np.convolve(reward_hist, np.ones(500)/500, 'valid'));
    plt.title("Reward (rolling 500)")
    plt.subplot(1,3,3); plt.plot(loss_hist); plt.title("Per‑step loss")
    plt.tight_layout(); plt.show()

    # optional: save the learned table
    agent.save_model(os.path.join(save_dir,
                     f"{agent.count_type}_q_state_learn_model.json"))

    return agent, env

    



if __name__ == "__main__":
    agent = QStateAgent(state_size = 4, action_size = 4)
    env = BlackjackEnv(count_type = "empty")

    train_q_agent(agent, env, episodes=100000, update_target_every=10, print_every=10000)
    evaluate_agent(agent, env, episodes=10000)
    model(agent)