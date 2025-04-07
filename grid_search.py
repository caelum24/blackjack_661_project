import itertools
from hyperparameters import HyperParameters
from train_agent import train_agent, evaluate_agent
from q_state_modeling import model

def perform_grid_search(deck_counts, gammas , batch_sizes, learning_rates, epsilon_starts, epsilon_decays, 
                        epsilon_mins, min_bets, max_bets, initial_bankrolls, memory_sizes):
    results = []
    #loops for hyperparameter grid search
    for deck_count, gamma, batch_size, lr, eps_start, eps_decay, eps_min, min_bet, max_bet, init_bankroll, mem_size in itertools.product(deck_counts, gammas, batch_sizes,
        learning_rates, epsilon_starts, epsilon_decays, epsilon_mins, min_bets, max_bets, initial_bankrolls, memory_sizes):
        HyperParameters.DECK_COUNT    = deck_count
        HyperParameters.GAMMA         = gamma
        HyperParameters.BATCH_SIZE    = batch_size
        HyperParameters.LEARNING_RATE = lr
        HyperParameters.EPSILON_START = eps_start
        HyperParameters.EPSILON_DECAY = eps_decay
        HyperParameters.EPSILON_MIN   = eps_min
        HyperParameters.MIN_BET       = min_bet
        HyperParameters.MAX_BET       = max_bet
        HyperParameters.INITIAL_BANKROLL = init_bankroll
        HyperParameters.MEMORY_SIZE   = mem_size

        print("\n=====================================")
        print(f"Training with: DECK_COUNT={deck_count}, GAMMA={gamma}, "
            f"BATCH_SIZE={batch_size}, LR={lr}, EPS_START={eps_start}, "
            f"EPS_DECAY={eps_decay}, EPS_MIN={eps_min}")
        print("=====================================\n")

        agent, env = train_agent(episodes=10000, update_target_every=200, print_every=2000)

        avg_reward = evaluate_agent(agent, env, episodes=2000)

        # Get accuracy vs. basic strategy
        try:
            hard_acc, soft_acc, pairs_acc = model(agent)
            accuracy_summary = (hard_acc + soft_acc + pairs_acc) / 3
        except:
            hard_acc, soft_acc, pairs_acc, accuracy_summary = None, None, None, None
        
        results.append({
            "DECK_COUNT":    deck_count,
            "GAMMA":         gamma,
            "BATCH_SIZE":    batch_size,
            "LEARNING_RATE": lr,
            "EPS_START":     eps_start,
            "EPS_DECAY":     eps_decay,
            "EPS_MIN":       eps_min,
            "AverageReward": avg_reward,
            "HardAcc":       hard_acc,
            "SoftAcc":       soft_acc,
            "PairsAcc":      pairs_acc,
            "OverallAcc":    accuracy_summary
        })

    print("\nGRID SEARCH COMPLETE")
    for r in results:
        print(r)
