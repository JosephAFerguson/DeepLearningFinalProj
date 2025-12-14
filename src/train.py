import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.lstm_gnn import DeepLSTMGNN
from utils.data_loader import CryptoGraphDataset
import matplotlib.pyplot as plt

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
WINDOW_SIZE = 60
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def train():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. Prepare Data
    dataset = CryptoGraphDataset(
        prices_path="../data/raw/crypto_prices.csv",
        global_path="../data/raw/global_metrics.csv", # <--- Add this argument
        adj_path="../data/processed/rolling_adj_matrices.pt",
        window_size=WINDOW_SIZE
    )
    
    # Split Train (80%) / Test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # NO random_split! Use Subset with ranges.
    train_data = torch.utils.data.Subset(dataset, range(0, train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    # 3. DataLoaders
    # shuffle=True is OK for train_loader (it shuffles batches WITHIN the training period)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # shuffle=False for test_loader (keep the timeline in order for plotting)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    # 2. Initialize Model
    # Get num_nodes from dataset (columns count)
    num_nodes = len(dataset.coin_names)
    
    model = DeepLSTMGNN(
        num_nodes=num_nodes,
        in_channels=6,      # Price + additional features
        hidden_dim=64       # Hiddn dimension.
    ).to(DEVICE)
    
    print(f"Model:\n")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #criterion = nn.MSELoss() 
    criterion = nn.HuberLoss(delta=1.0) # <-- USE THIS
    print("\n Starting Training")
    print(f"-" * 75)
    # 3. Training Loop
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, adj, y) in enumerate(train_loader):
            x, adj, y = x.to(DEVICE), adj.to(DEVICE), y.to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            predictions = model(x, adj)
            
            # Loss Calculation
            loss = criterion(predictions, y)
            
            # Backward
            loss.backward()
            # graidnt clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # 4. Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, adj, y in test_loader:
                x, adj, y = x.to(DEVICE), adj.to(DEVICE), y.to(DEVICE)
                preds = model(x, adj)
                v_loss = criterion(preds, y)
                val_loss += v_loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)

    # 5. Save Model
    torch.save(model.state_dict(), "../src/models/lstm_gnn_model.pth")
    print("Model saved!")
    
    # 6. Plot Training Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()