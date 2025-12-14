import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.lstm_baseline import VanillaLSTM
from utils.data_loader import CryptoGraphDataset
import matplotlib.pyplot as plt

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 0.0001
WINDOW_SIZE = 60
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def train_baseline():
    print(f"--- Starting BASELINE Training on {DEVICE} ---")
    
    # 1. Prepare Data (Updated for 6 Features & Macro Data)
    dataset = CryptoGraphDataset(
        prices_path="../data/raw/crypto_prices.csv",
        global_path="../data/raw/global_metrics.csv", # <--- Added this
        adj_path="../data/processed/rolling_adj_matrices.pt",
        window_size=WINDOW_SIZE,
        train_ratio=0.8
    )
    
    # --- FIX: SEQUENTIAL SPLIT (Crucial) ---
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Use Subset to slice sequentially (0 to 80% = Train, 80% to 100% = Test)
    train_data = torch.utils.data.Subset(dataset, range(0, train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    # 2. Initialize Vanilla LSTM
    # It must take 6 features now!
    model = VanillaLSTM(
        in_channels=6,      # <--- CHANGED TO 6 (Price, Vol, Ret, Risk, Dom, MktCap)
        hidden_dim=64
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    
    # 3. Train Loop
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x, adj, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Note: Baseline ignores 'adj'
            
            optimizer.zero_grad()
            predictions = model(x, adj) 
            loss = criterion(predictions, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Added clipping for stability
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, adj, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x, adj)
                val_loss += criterion(preds, y).item()
        
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)

    # 4. Save
    torch.save(model.state_dict(), "../src/models/lstm_baseline.pth")
    print("Baseline Model saved!")
    
    # Plot comparison
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Baseline Training Progress')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_baseline()