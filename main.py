'''
Master file for the blackjack agent.
'''
import argparse
import torch
# from train_agent import train_agent
from split_train_agent import train_agent
from modeling import model
from DQNAgent import DQNAgent
# from environment import BlackjackEnv
from split_environment import BlackjackEnv

def load_model(model_path, count_type="empty"):
    """Load a trained model from file"""
    env = BlackjackEnv(count_type="count_type")
    state_size = 4  # player_sum, dealer_up_card, usable_ace, can_double
    action_size = 3  # hit, stand, double
    
    # Create agent with same architecture
    agent = DQNAgent(count_type="count_type")
    
    # Load saved state
    checkpoint = torch.load(model_path)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    
    print(f"Loaded model from {model_path}")
    print(f"Model epsilon: {agent.epsilon}")
    
    return agent, env

def main():
    parser = argparse.ArgumentParser(description='Blackjack DQN Agent')
    parser.add_argument('--load', type=str, help='Path to load a trained model')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train (default: 10000)')
    parser.add_argument('--count_type', type=str, default='empty', choices=['empty', 'full', 'hi_lo', 'zen', 'uston_apc', 'ten_count'],
                      help='Card counting system to use (default: empty)')
    args = parser.parse_args()

    if args.load:
        # Load existing model
        agent, env = load_model(args.load, count_type=args.count_type)
    else:
        # Train new model
        print(f"Training new model for {args.episodes} episodes with {args.count_type} counting system...")
        agent = DQNAgent(args.count_type)
        agent, env = train_agent(agent, episodes=args.episodes, print_every=100)

    # Evaluate and visualize the model
    model(agent)

if __name__ == "__main__":
    main()