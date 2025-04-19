from train_betting_agent import load_and_run_model
from train_betting_agent import train_betting_agent_from_run_data
from train_betting_agent import evaluate_agent
import torch
from main import load_model

# Load playing agent
playing_agent = DQNAgent("full")
playing_agent.load_model("models/blackjack_agent_full_finished.pth")
print("Model loaded from models/blackjack_agent_full_finished.pth")

# Set up environment
env = BlackjackEnv(count_type="full")

# Load dictionary of count states and rewards
with open("count_reward_dict.pkl", "rb") as f:
    dictionary = pickle.load(f)

count_type = "full"
batch_size = 128

# Load previous checkpoint
checkpoint_path = "checkpoints/full_20250418_135645_continued/final_model.pt"
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    loaded_model, checkpoint_data = load_checkpoint(checkpoint_path)
    
    # Continue training from checkpoint - pass the loaded model to the training function
    print(f"Continuing training for 100 more epochs...")
    model = train_betting_nn(
        dictionary, 
        count_type, 
        batch_size, 
        epochs=200,
        sample_size=10000,
        checkpoint_dir="checkpoints",
        initial_model=loaded_model,
        initial_checkpoint=checkpoint_data
    )
else:
    print(f"Checkpoint {checkpoint_path} not found, training new model")
    model = train_betting_nn(dictionary, count_type, batch_size, epochs=100)

# Evaluate the trained model
evaluate_nn_betting_agent(model, playing_agent, env, count_type='full', episodes=1000, max_bet=1000, min_bet=1)

print("Betting agent trained and saved.")