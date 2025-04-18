from train_betting_agent import load_and_run_model
from train_betting_agent import train_betting_agent_from_run_data
from train_betting_agent import evaluate_agent
import torch
from main import load_model
import argparse
from split_train_agent import train_agent
from modeling import model
from DQNAgent import DQNAgent
from split_environment import BlackjackEnv
from Supervised_betting import collect_dictionary
from Supervised_betting import evaluate_nn_betting_agent
from BettingModel import BettingNN
from BettingModel import train_betting_nn
import pickle

playing_agent = DQNAgent("full")
# print(f"Training new model for {1000} episodes...")
# playing_agent, env = train_agent(agent, episodes=100000, print_every=100)
playing_agent.load_model("models/blackjack_agent_checkpoint_finished_model18000.pth")
env = BlackjackEnv(count_type="full")
dictionary = collect_dictionary(playing_agent, env, count_type="full", num_episodes=1000000)
with open("count_reward_dict.pkl", "wb") as f:
    pickle.dump(dictionary, f)

#with open("count_reward_dict_10mil.pkl", "rb") as f:
    #dictionary = pickle.load(f)
count_type = "full"
batch_size = "32"
model = train_betting_nn(dictionary, count_type, batch_size, epochs=10000)

#run_data = load_and_run_model(playing_agent, env, num_episodes=50000)
#possible_bets = [1, 2, 3, 4, 5, 10, 25, 50, 100]
#betting_agent, loss_history = train_betting_agent_from_run_data(run_data, possible_bets, state_dim=2, num_training_steps=10000)
#evaluate_agent(betting_agent, playing_agent, env, episodes=1000)
#torch.save(betting_agent.policy_net.state_dict(), "betting_agent.pth")

evaluate_nn_betting_agent(model, playing_agent, env, episodes = 1000, max_bet = 1000, min_bet = 1)


print("Betting agent trained and saved.")