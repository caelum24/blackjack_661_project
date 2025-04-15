from train_betting_agent import load_and_run_model
from train_betting_agent import train_betting_agent_from_run_data
from train_betting_agent import evaluate_agent
import torch
from main import load_model
import argparse
# from train_agent import train_agent
from split_train_agent import train_agent
from modeling import model
from DQNAgent import DQNAgent
# from environment import BlackjackEnv
from split_environment import BlackjackEnv

agent = DQNAgent("empty")
print(f"Training new model for {1000} episodes...")
playing_agent, env = train_agent(agent, episodes=10000, print_every=100)


run_data = load_and_run_model(playing_agent, env, num_episodes=5000)

possible_bets = [1, 5, 10, 25, 50, 100]
betting_agent, loss_history = train_betting_agent_from_run_data(run_data, possible_bets, state_dim=2, num_training_steps=100000)

evaluate_agent(betting_agent, playing_agent, env, episodes=1000)

torch.save(betting_agent.policy_net.state_dict(), "betting_agent.pth")
print("Betting agent trained and saved.")