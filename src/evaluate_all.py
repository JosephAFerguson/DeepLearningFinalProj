import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import all models
from models.lstm_gnn import DeepLSTMGNN, DeepBiLSTMGNN
from models.lstm_baseline import VanillaLSTM
from utils.data_loader import CryptoGraphDataset

# --- Config ---
GNN_PATH = "../src/models/lstm_gnn_model.pth"
BASELINE_PATH = "../src/models/lstm_baseline.pth"
BIGNN_PATH = "../src/models/bi_lstm_gnn_model.pth" # Check your filename!
DATA_PATH = "../data/raw/crypto_prices.csv"
GLOBAL_PATH = "../data/raw/global_metrics.csv"
ADJ_PATH = "../data/processed/rolling_adj_matrices.pt"
METRICS_OUTPUT_PATH = "../data/processed/model_metrics.csv"
WINDOW_SIZE = 60

# Auto-detect device
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

def calculate_metrics(actuals, preds):
    """Computes RMSE, MAE, MAPE, and Directional Accuracy."""
    if len(actuals) == 0: return {}
    
    # 1. Basic Error Metrics
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    
    # 2. MAPE
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100 if np.sum(mask) > 0 else 0.0
    
    # 3. Directional Accuracy
    actual_delta = np.diff(actuals)
    pred_delta = np.diff(preds)
    direction_matches = np.sign(actual_delta) == np.sign(pred_delta)
    directional_acc = np.mean(direction_matches) * 100
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Dir_Acc": directional_acc
    }

def run_evaluation(mode='all', target_coin=None):
    print(f"\n--- Starting Evaluation (Mode: {mode}) ---")
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
    
    num_nodes = len(dataset.coin_names)
    print(f"Loaded dataset with {num_nodes} coins.")
    
    # --- 2. Load Models ---

    # A. LSTM-GNN
    model_gnn = DeepLSTMGNN(num_nodes=num_nodes, in_channels=6, hidden_dim=64).to(DEVICE)
    try:
        model_gnn.load_state_dict(torch.load(GNN_PATH, map_location=DEVICE))
        model_gnn.eval()
        print("[+] LSTM-GNN Model loaded.")
    except Exception as e:
        model_gnn = None
        print(f"[-] LSTM-GNN Model NOT found or error: {e}")

    # B. Baseline LSTM
    model_base = VanillaLSTM(in_channels=6, hidden_dim=64).to(DEVICE) # Ensure baseline hidden_dim matches training
    try:
        model_base.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
        model_base.eval()
        print("[+] Baseline Model loaded.")
    except Exception as e:
        model_base = None
        print(f"[-] Baseline Model NOT found or error: {e}")
    
    # C. Bi-LSTM-GNN (FIXED LOADING LOGIC)
    # Fix 1: Added num_nodes argument
    model_bilstmgnn = DeepBiLSTMGNN(num_nodes=num_nodes, in_channels=6, hidden_dim=64).to(DEVICE)
    try:
        model_bilstmgnn.load_state_dict(torch.load(BIGNN_PATH, map_location=DEVICE))
        model_bilstmgnn.eval() # Fix 2: Set CORRECT model to eval
        print("[+] Bi-LSTM-GNN Model loaded.")
    except Exception as e:
        model_bilstmgnn = None
        print(f"[-] Bi-LSTM-GNN Model NOT found or error: {e}")

    # Check if at least one model exists
    if not model_gnn and not model_base and not model_bilstmgnn:
        print("Error: No models found!")
        return

    # 3. Generate Predictions
    print("Generating predictions...")
    
    all_actuals = []
    all_preds_gnn = []
    all_preds_base = []
    all_preds_bignn = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for x, adj, y in loader:
            x, adj = x.to(DEVICE), adj.to(DEVICE)
            
            all_actuals.append(y.squeeze(-1).cpu().numpy())
            
            if model_gnn:
                out = model_gnn(x, adj)
                all_preds_gnn.append(out.squeeze(-1).cpu().numpy())
                
            if model_base:
                out = model_base(x, adj)
                all_preds_base.append(out.squeeze(-1).cpu().numpy())
            
            if model_bilstmgnn:
                out = model_bilstmgnn(x, adj)
                all_preds_bignn.append(out.squeeze(-1).cpu().numpy())

    # Concatenate
    real_actuals = np.concatenate(all_actuals, axis=0)
    
    # Inverse scaling
    print("Inverse scaling...")
    real_actuals = dataset.scalers[0].inverse_transform(real_actuals)
    
    real_gnn = None
    if all_preds_gnn:
        raw_gnn = np.concatenate(all_preds_gnn, axis=0)
        real_gnn = dataset.scalers[0].inverse_transform(raw_gnn)
        
    real_base = None
    if all_preds_base:
        raw_base = np.concatenate(all_preds_base, axis=0)
        real_base = dataset.scalers[0].inverse_transform(raw_base)
    
    real_bignn = None
    if all_preds_bignn:
        raw_bignn = np.concatenate(all_preds_bignn, axis=0)
        real_bignn = dataset.scalers[0].inverse_transform(raw_bignn)

    # 4. Mode A: Export CSV
    if mode == 'all':
        print(f"\nComputing metrics for all {num_nodes} coins...")
        metrics_list = []
        
        for i, coin in enumerate(dataset.coin_names):
            row = {'Symbol': coin}
            
            # Baseline Metrics
            if real_base is not None:
                m = calculate_metrics(real_actuals[:, i], real_base[:, i])
                row.update({
                    'Base_RMSE': m['RMSE'], 
                    'Base_MAE': m['MAE'], 
                    'Base_MAPE': m['MAPE'], 
                    'Base_DirAcc': m['Dir_Acc']
                })
            
            # GNN Metrics
            if real_gnn is not None:
                m = calculate_metrics(real_actuals[:, i], real_gnn[:, i])
                row.update({
                    'GNN_RMSE': m['RMSE'], 
                    'GNN_MAE': m['MAE'], 
                    'GNN_MAPE': m['MAPE'], 
                    'GNN_DirAcc': m['Dir_Acc']
                })
            
            # Bi-GNN Metrics
            if real_bignn is not None:
                m = calculate_metrics(real_actuals[:, i], real_bignn[:, i])
                row.update({
                    'BIGNN_RMSE': m['RMSE'], 
                    'BIGNN_MAE': m['MAE'], 
                    'BIGNN_MAPE': m['MAPE'], 
                    'BIGNN_DirAcc': m['Dir_Acc']
                })
            metrics_list.append(row)
            
        df = pd.DataFrame(metrics_list)
        df.to_csv(METRICS_OUTPUT_PATH, index=False)
        print(f"SUCCESS: Metrics exported to {METRICS_OUTPUT_PATH}")

    # 5. Mode B: Plot Specific Coin
    elif mode == 'plot' and target_coin:
        try:
            idx = dataset.coin_names.index(target_coin)
            print(f"\nPlotting {target_coin}...")
            
            plt.figure(figsize=(14, 7))
            plt.plot(real_actuals[:, idx], label="Actual", color='black', alpha=0.6)
            
            if real_base is not None:
                plt.plot(real_base[:, idx], label="Baseline", color='orange', linestyle='--')
            
            if real_gnn is not None:
                plt.plot(real_gnn[:, idx], label="LSTM-GNN", color='cyan')
                
            if real_bignn is not None:
                # Fix 3: Changed color to magenta for visibility
                plt.plot(real_bignn[:, idx], label="Bi-LSTM-GNN", color='magenta', linestyle='-.')
                
            plt.title(f"{target_coin} Price Prediction Comparison")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ValueError:
            print(f"Coin {target_coin} not found.")

if __name__ == "__main__":
    # Example: Plot BTC or run 'all'
    # run_evaluation('plot', 'BTC')
    run_evaluation('all')