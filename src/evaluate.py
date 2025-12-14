import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import ALL models (Baseline, GNN, and Bi-GNN)
from models.lstm_gnn import DeepLSTMGNN, DeepBiLSTMGNN
from models.lstm_baseline import VanillaLSTM
from utils.data_loader import CryptoGraphDataset

# --- Config ---
GNN_PATH = "../src/models/lstm_gnn_model.pth"
BASELINE_PATH = "../src/models/lstm_baseline.pth"
BIGNN_PATH = "../src/models/bi_lstm_gnn_model.pth" # <--- Path to new model
DATA_PATH = "../data/raw/crypto_prices.csv"
GLOBAL_PATH = "../data/raw/global_metrics.csv"
ADJ_PATH = "../data/processed/rolling_adj_matrices.pt"
WINDOW_SIZE = 60

# Auto-detect device
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def calculate_metrics(actuals, preds):
    """
    Computes RMSE, MAE, MAPE, and Directional Accuracy.
    """
    if len(actuals) == 0: return {}
    
    # 1. Basic Error Metrics
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    
    # 2. MAPE (Avoid division by zero)
    mask = actuals != 0
    if np.sum(mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100
    
    # 3. Directional Accuracy
    actual_delta = np.diff(actuals)
    pred_delta = np.diff(preds)
    direction_matches = np.sign(actual_delta) == np.sign(pred_delta)
    directional_acc = np.mean(direction_matches) * 100
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Directional Accuracy": directional_acc
    }

def evaluate(target_coin_symbol="BTC"):
    print(f"\n--- Starting Professional Evaluation for {target_coin_symbol} ---")
    print(f"Device: {DEVICE}")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("Error: Data file not found.")
        return

    dataset = CryptoGraphDataset(
        prices_path=DATA_PATH,
        global_path=GLOBAL_PATH,
        adj_path=ADJ_PATH,
        window_size=WINDOW_SIZE,
        train_ratio=0.8
    )
    
    try:
        coin_idx = dataset.coin_names.index(target_coin_symbol)
    except ValueError:
        print(f"Error: {target_coin_symbol} not found. Options: {dataset.coin_names}")
        return

    num_nodes = len(dataset.coin_names)
    print(f"Evaluating index {coin_idx} out of {num_nodes} coins.")
    
    # --- 2. Load Models ---
    
    # A. LSTM-GNN
    model_gnn = DeepLSTMGNN(num_nodes=num_nodes, in_channels=6, hidden_dim=64).to(DEVICE)
    try:
        model_gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
        model_gnn.eval()
        print("[+] LSTM-GNN Model loaded.")
    except FileNotFoundError:
        print("[-] GNN Model file not found. Skipping.")
        model_gnn = None

    # B. Baseline LSTM
    model_base = VanillaLSTM(in_channels=6, hidden_dim=64).to(DEVICE)
    try:
        model_base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        model_base.eval()
        print("[+] Baseline Model loaded.")
    except FileNotFoundError:
        print("[-] Baseline Model file not found. Skipping.")
        model_base = None

    # C. Bidirectional LSTM-GNN (New)
    model_bignn = DeepBiLSTMGNN(num_nodes=num_nodes, in_channels=6, hidden_dim=64).to(DEVICE)
    try:
        model_bignn.load_state_dict(torch.load(BIGNN_PATH, map_location=DEVICE))
        model_bignn.eval()
        print("[+] Bi-LSTM-GNN Model loaded.")
    except FileNotFoundError:
        print("[-] Bi-LSTM-GNN Model file not found. Skipping.")
        model_bignn = None

    if not model_gnn and not model_base and not model_bignn:
        print("Error: No models found! Train them first.")
        return

    # 3. Generate Predictions
    print("Generating predictions...")
    preds_gnn = []
    preds_base = []
    preds_bignn = []
    actuals = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for x, adj, y in loader:
            x, adj = x.to(DEVICE), adj.to(DEVICE)
            
            # Actuals
            target_actuals = y[:, coin_idx, 0].cpu().numpy()
            actuals.extend(target_actuals)
            
            # GNN
            if model_gnn:
                out_gnn = model_gnn(x, adj)
                target_gnn = out_gnn[:, coin_idx, 0].cpu().numpy()
                preds_gnn.extend(target_gnn)
                
            # Baseline
            if model_base:
                out_base = model_base(x, adj)
                target_base = out_base[:, coin_idx, 0].cpu().numpy()
                preds_base.extend(target_base)
            
            # Bi-GNN
            if model_bignn:
                out_bignn = model_bignn(x, adj)
                target_bignn = out_bignn[:, coin_idx, 0].cpu().numpy()
                preds_bignn.extend(target_bignn)

    # 4. Inverse Transform
    def inverse_scale(data_list):
        if not data_list: return np.array([])
        arr = np.array(data_list)
        # Create dummy matrix (Samples, Num_Coins)
        dummy = np.zeros((len(arr), num_nodes))
        # Fill target column
        dummy[:, coin_idx] = arr
        # Inverse transform
        rescaled_matrix = dataset.scalers[0].inverse_transform(dummy)
        return rescaled_matrix[:, coin_idx]

    print("Inverse scaling data...")
    real_actuals = inverse_scale(actuals)
    real_gnn = inverse_scale(preds_gnn)
    real_base = inverse_scale(preds_base)
    real_bignn = inverse_scale(preds_bignn)

    # 5. Calculate Metrics
    print(f"\n--- Final Evaluation Metrics for {target_coin_symbol} ---")
    
    if len(real_base) > 0:
        metrics = calculate_metrics(real_actuals, real_base)
        print(f"\nBASELINE (Vanilla LSTM):")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE:  ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")
        print(f"  Dir Acc: {metrics['Directional Accuracy']:.2f}%")

    if len(real_gnn) > 0:
        metrics = calculate_metrics(real_actuals, real_gnn)
        print(f"\nPROPOSED (LSTM-GNN):")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE:  ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")
        print(f"  Dir Acc: {metrics['Directional Accuracy']:.2f}%")
    
    if len(real_bignn) > 0:
        metrics = calculate_metrics(real_actuals, real_bignn)
        print(f"\nPROPOSED (Bi-LSTM-GNN):")
        print(f"  RMSE: ${metrics['RMSE']:.2f}")
        print(f"  MAE:  ${metrics['MAE']:.2f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}%")
        print(f"  Dir Acc: {metrics['Directional Accuracy']:.2f}%")

    # 6. Plotting - Zoomed In (Recent)
    plt.figure(figsize=(14, 7))
    start_idx = 1500 if len(real_actuals) > 1500 else 0
    
    plt.plot(real_actuals[start_idx:], label="Actual Price", color='black', linewidth=2, alpha=0.6)
    
    if len(real_base) > 0:
        plt.plot(real_base[start_idx:], label="Baseline", color='magenta', linestyle='--', linewidth=1.5)
    #if len(real_gnn) > 0:
        #plt.plot(real_gnn[start_idx:], label="LSTM-GNN", color='cyan', linewidth=1.5)
    if len(real_bignn) > 0:
        plt.plot(real_bignn[start_idx:], label="Bi-LSTM-GNN", color='aqua', linewidth=1.5, linestyle='-.')

    plt.title(f"Comparison (Recent): {target_coin_symbol}")
    plt.xlabel("Recent Time (Days)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"../data/processed/eval_professional_{target_coin_symbol}_recent.png"
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    plt.show()
    
    # 7. Plotting - Full History
    plt.figure(figsize=(14, 7))
    plt.plot(real_actuals, label="Actual Price", color='black', linewidth=2, alpha=0.6)
    
    if len(real_base) > 0:
        plt.plot(real_base, label="Vanilla LSTM (Baseline)", color='magenta', linestyle='--', linewidth=1.5)
    #if len(real_gnn) > 0:
        #plt.plot(real_gnn, label="LSTM-GNN", color='blue', linewidth=1.5)
    #if len(real_bignn) > 0:
        plt.plot(real_bignn, label="Bi-LSTM-GNN", color='aqua', linewidth=1.5, linestyle='-.')

    plt.title(f"Full History Prediction: {target_coin_symbol}")
    plt.xlabel("Time (Days)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"../data/processed/eval_professional_{target_coin_symbol}_full.png"
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate("XMR")