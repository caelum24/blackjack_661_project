from train_betting_agent import load_and_run_model
from train_betting_agent import train_betting_agent_from_run_data
from train_betting_agent import evaluate_agent
import torch
from main import load_model


run_data = load_and_run_model("models/blackjack_agent_20250407_185346.pth", num_episodes=5000)

possible_bets = [1, 5, 10, 25, 50, 100]
betting_agent, loss_history = train_betting_agent_from_run_data(run_data, possible_bets)

playing_agent, env = load_model("models/blackjack_agent_20250407_185346.pth")

evaluate_agent(betting_agent, playing_agent, env, episodes=1000)

torch.save(betting_agent.policy_net.state_dict(), "betting_agent.pth")
print("Betting agent trained and saved.")