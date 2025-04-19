import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import datetime

class BettingNN(nn.Module):
    def __init__(self, input_dim=3):
        super(BettingNN, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )

        # Wider network with batch normalization
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Since you're predicting probability
        )

    def forward(self, x):
        return self.model(x)


def train_betting_nn(dictionary, count_type, batch_size, epochs=100, sample_size=10000, checkpoint_dir="checkpoints", 
                   initial_model=None, initial_checkpoint=None):
    """
    Train the betting neural network with sampling from a large dictionary
    
    Args:
        dictionary: Dictionary with a million keys
        count_type: Type of counting system used
        batch_size: Size of training batches
        epochs: Number of training epochs
        sample_size: Number of samples to use per epoch (much smaller than full dictionary)
        checkpoint_dir: Directory to save checkpoints
        initial_model: Optional pre-trained model to continue training
        initial_checkpoint: Optional checkpoint data from previous training
    """
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create unique run ID using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{count_type}_{timestamp}"
    
    # If continuing from checkpoint, add a suffix
    if initial_model is not None:
        run_id += "_continued"
        
    run_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_dir)
    
    print(f"Starting training run: {run_id}")
    print(f"Checkpoints will be saved to: {run_dir}")
    
    print_every = max(1, epochs//10)
    checkpoint_every = max(1, epochs//20)  # Save checkpoint every 5% of total epochs

    counting_type_state_size = {"full":10, "hi_lo":1, "zen":1, "uston_apc":1, "ten_count":1, "comb_counts":4}

    if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count", "comb_counts"]:
        print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count", "comb_counts"])
        raise ValueError 
    state_size = counting_type_state_size[count_type]

    # dataset = CountRewardDataset(dictionary)
    # dataloader = DataLoader(dataset, batch_size, shuffle=True)

    # model = BettingNN(state_size)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # for epoch in range(epochs):
    #     epoch_loss = 0.0
    #     for inputs, labels in dataloader:

    # Use learning rate scheduler and different loss function
    if initial_model is not None:
        model = initial_model
        print("Continuing training from provided model")
    else:
        model = BettingNN(state_size)
        print("Created new model")
        
    criterion = nn.BCELoss()  # Binary cross entropy for probability prediction
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    
    # If we have a checkpoint, load optimizer and scheduler states
    start_epoch = 0
    if initial_checkpoint is not None and 'optimizer_state_dict' in initial_checkpoint:
        optimizer.load_state_dict(initial_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(initial_checkpoint['scheduler_state_dict'])
        start_epoch = initial_checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    # Track best model
    best_loss = float('inf')
    if initial_checkpoint is not None and 'val_loss' in initial_checkpoint:
        best_loss = initial_checkpoint['val_loss']
        print(f"Previous best validation loss: {best_loss:.4f}")
        
    best_model_state = None
    patience_counter = 0
    patience_limit = 100  # Early stopping patience
    
    # Save training config
    config = {
        "count_type": count_type,
        "batch_size": batch_size,
        "epochs": epochs,
        "sample_size": sample_size,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "weight_decay": 1e-5,
        "state_size": state_size,
        "dataset_size": len(dictionary),
        "continued_from_checkpoint": initial_model is not None,
        "starting_epoch": start_epoch
    }
    
    # Save config to file
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Create full dataset once
    full_dataset = CountRewardDataset(dictionary)
    
    # Add validation set
    dataset_size = len(full_dataset)
    val_size = min(10000, int(dataset_size * 0.1))
    indices = torch.randperm(dataset_size)
    val_indices = indices[:val_size]
    
    # Initialize or continue training history
    if initial_checkpoint is not None and 'history' in initial_checkpoint:
        history = initial_checkpoint['history']
        print(f"Continuing training history with {len(history['train_loss'])} previous epochs")
    else:
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
    
    for epoch in range(epochs):
        actual_epoch = start_epoch + epoch
        
        # Sample training indices (excluding validation indices)
        train_indices = [idx for idx in indices[val_size:] if idx not in val_indices]
        train_indices = train_indices[:sample_size]
        
        # Create datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size)
        
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    #         epoch_loss += loss.item() * inputs.size(0)
    #     avg_loss = epoch_loss / len(dataset)
    #     if epoch % print_every == 0:
    #         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # return model

            train_loss += loss.item() * inputs.size(0)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        
        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduler step
        scheduler.step(avg_val_loss)
        
        # Print progress
        if epoch % print_every == 0:
            print(f"Epoch {actual_epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save regular checkpoint
        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{actual_epoch+1}.pt")
            torch.save({
                'epoch': actual_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {actual_epoch+1}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model checkpoint
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save({
                'epoch': actual_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history
            }, best_model_path)
            print(f"Saved new best model at epoch {actual_epoch+1} with val_loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {actual_epoch+1}")
            break
    
    # Save final model
    final_model_path = os.path.join(run_dir, "final_model.pt")
    torch.save({
        'epoch': actual_epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'history': history
    }, final_model_path)
    print(f"Saved final model after epoch {actual_epoch+1}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_loss:.4f}")
    
    return model

# Also add a function to load checkpoints
def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary with training state
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Extract state size from the first layer
    first_layer = next(iter(checkpoint['model_state_dict'].items()))
    if 'weight' in first_layer[0]:
        input_dim = first_layer[1].shape[1]
    else:
        # Try the second item which should be weights
        first_layer = list(checkpoint['model_state_dict'].items())[1]
        input_dim = first_layer[1].shape[1]
    
    model = BettingNN(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}, Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, checkpoint
class CountRewardDataset(Dataset):
    def __init__(self, count_reward_dict):
        self.inputs = []
        self.labels = []

        for count_state, (avg_reward, _) in count_reward_dict.items():
            input_tensor = torch.tensor(count_state, dtype=torch.float32)
            # Convert avg_reward to be compatible with sigmoid output (0-1)
            label_tensor = torch.tensor([avg_reward], dtype=torch.float32)
            self.inputs.append(input_tensor)
            self.labels.append(label_tensor)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]