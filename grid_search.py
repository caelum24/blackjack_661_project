import itertools
import csv
from datetime import datetime
import torch
import gc
from hyperparameters import HyperParameters
from train_agent import train_agent, evaluate_agent
from q_state_modeling import model

def perform_grid_search(deck_counts, gammas, batch_sizes, learning_rates, epsilon_starts, epsilon_decays, 
                        epsilon_mins, min_bets, max_bets, initial_bankrolls, memory_sizes):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"grid_search_results_{timestamp}.csv"
    
    # Create CSV file and write header
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["DECK_COUNT", "GAMMA", "BATCH_SIZE", "LEARNING_RATE", "EPS_START", 
                     "EPS_DECAY", "EPS_MIN", "MIN_BET", "MAX_BET", "INITIAL_BANKROLL", 
                     "MEMORY_SIZE", "AverageReward", "HardAcc", "SoftAcc", "PairsAcc", "OverallAcc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Calculate total combinations for progress tracking
        total_combinations = len(deck_counts) * len(gammas) * len(batch_sizes) * len(learning_rates) * \
                           len(epsilon_starts) * len(epsilon_decays) * len(epsilon_mins) * \
                           len(min_bets) * len(max_bets) * len(initial_bankrolls) * len(memory_sizes)
        current_combination = 0
        
        #loops for hyperparameter grid search
        for deck_count, gamma, batch_size, lr, eps_start, eps_decay, eps_min, min_bet, max_bet, init_bankroll, mem_size in itertools.product(
            deck_counts, gammas, batch_sizes, learning_rates, epsilon_starts, epsilon_decays, 
            epsilon_mins, min_bets, max_bets, initial_bankrolls, memory_sizes):
            
            current_combination += 1
            print(f"\nProgress: {current_combination}/{total_combinations} combinations")
            
            # Update hyperparameters
            HyperParameters.DECK_COUNT = deck_count
            HyperParameters.GAMMA = gamma
            HyperParameters.BATCH_SIZE = batch_size
            HyperParameters.LEARNING_RATE = lr
            HyperParameters.EPSILON_START = eps_start
            HyperParameters.EPSILON_DECAY = eps_decay
            HyperParameters.EPSILON_MIN = eps_min
            HyperParameters.MIN_BET = min_bet
            HyperParameters.MAX_BET = max_bet
            HyperParameters.INITIAL_BANKROLL = init_bankroll
            HyperParameters.MEMORY_SIZE = mem_size

            print("\n=====================================")
            print(f"Training with: DECK_COUNT={deck_count}, GAMMA={gamma}, "
                f"BATCH_SIZE={batch_size}, LR={lr}, EPS_START={eps_start}, "
                f"EPS_DECAY={eps_decay}, EPS_MIN={eps_min}")
            print("=====================================\n")

            # Train and evaluate
            agent, env = train_agent(episodes=10000, update_target_every=200, print_every=2000)
            avg_reward = evaluate_agent(agent, env, episodes=2000)

            # Get accuracy vs. basic strategy
            try:
                hard_acc, soft_acc, pairs_acc = model(agent)
                accuracy_summary = (hard_acc + soft_acc + pairs_acc) / 3
            except Exception as e:
                print(f"Error in model evaluation: {e}")
                hard_acc, soft_acc, pairs_acc, accuracy_summary = None, None, None, None
            
            # Create result dictionary
            result = {
                "DECK_COUNT": deck_count,
                "GAMMA": gamma,
                "BATCH_SIZE": batch_size,
                "LEARNING_RATE": lr,
                "EPS_START": eps_start,
                "EPS_DECAY": eps_decay,
                "EPS_MIN": eps_min,
                "MIN_BET": min_bet,
                "MAX_BET": max_bet,
                "INITIAL_BANKROLL": init_bankroll,
                "MEMORY_SIZE": mem_size,
                "AverageReward": avg_reward,
                "HardAcc": hard_acc,
                "SoftAcc": soft_acc,
                "PairsAcc": pairs_acc,
                "OverallAcc": accuracy_summary
            }
            
            # Write result to CSV file immediately
            writer.writerow(result)
            csvfile.flush()  # Force write to disk
            
            # Print current result
            print(f"\nCurrent Result: {result}")
            
            # Clear memory
            del agent
            del env
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

    print(f"\nGRID SEARCH COMPLETE. Results saved to {filename}")
    return filename
