import itertools
import csv
import pickle
from datetime import datetime
import torch
import gc
from DQNAgent import DQNAgent
from split_environment import BlackjackEnv
from Supervised_betting import collect_dictionary, evaluate_nn_betting_agent
from BettingModel import train_betting_nn

def betting_grid_search(playing_agent, env, count_types, dict_sizes, train_epochs, batch_sizes, episodes=2000, 
                        max_bet=1000, min_bet=1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"betting_grid_results_{timestamp}.csv"
    best_profit = float("-inf")
    best_model_path = "best_betting_model.pth"

    # Open CSV and write header
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ["CountType", "DictSize", "Epochs", "BatchSize", "AvgProfit"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for count_type, dict_size, epochs, batch_size in itertools.product(
                count_types, dict_sizes, train_epochs, batch_sizes):

            print(f"Grid Search: count={count_type}, dict={dict_size}, epochs={epochs}, batch={batch_size}")

            dictionary = collect_dictionary(playing_agent, env, count_type, dict_size)

            betting_model = train_betting_nn(dictionary, count_type, batch_size=batch_size, epochs=epochs)

            avg_profit = evaluate_nn_betting_agent(
                betting_model, playing_agent, env, count_type,
                episodes=episodes, max_bet=max_bet, min_bet=min_bet
            )

            writer.writerow({
                "CountType": count_type,
                "DictSize": dict_size,
                "Epochs": epochs,
                "BatchSize": batch_size,
                "AvgProfit": avg_profit
            })
            csvfile.flush()
            print(f"Result: AvgProfit={avg_profit:.4f}")

            if avg_profit > best_profit:
                best_profit = avg_profit
                torch.save(betting_model.state_dict(), best_model_path)
                print("New best betting model saved.")
            del betting_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Grid search complete. Results in {results_file}. Best profit: {best_profit:.4f}")
    return results_file
