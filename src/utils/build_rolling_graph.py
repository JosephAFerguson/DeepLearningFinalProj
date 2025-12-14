import pandas as pd
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt 

# --- CONFIG ---
INPUT_PATH = "../../data/raw/crypto_prices.csv"
OUTPUT_PATH = "../../data/processed/rolling_adj_matrices.pt"
PLOT_PATH = "../../data/processed/correlation_heatmap.png"

# Hyperparameters
WINDOW_SIZE = 60  # Lookback window for correlation
THRESHOLD = 0.3   # Lower threshold since we are using weights

def build_rolling_graph():
    print(f"\n--- Building Weighted Rolling Graph (Window: {WINDOW_SIZE}) ---")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return
    
    # 1. Load Data
    df = pd.read_csv(INPUT_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # --- FIX: REMOVE DUPLICATES ---
    # If fetch script ran twice, we might have duplicate rows. 
    # We drop them here to ensure (Date, Symbol) is unique.
    initial_len = len(df)
    df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
    
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} duplicate rows from CSV.")

    df = df.sort_values(['coin_id', 'date'])
    
    # 2. Pivot to (Date x Coin) matrix
    pivot_df = df.pivot(index='date', columns='symbol', values='price')
    
    # Handle missing data
    pivot_df = pivot_df.ffill().bfill()
    coin_names = pivot_df.columns.tolist()
    
    # 3. Calculate Log Returns (Match Data Loader logic)
    # ln(Pt / Pt-1)
    eps = 1e-8
    log_ret_df = np.log(pivot_df + eps) - np.log(pivot_df.shift(1) + eps)
    log_ret_df = log_ret_df.fillna(0)
    
    print(f"Data Shape: {log_ret_df.shape} (Days x Coins)")
    
    # 4. Rolling Correlation Loop
    num_days = len(log_ret_df)
    num_coins = len(coin_names)
    
    adj_matrices = []
    
    print("Computing rolling weighted matrices...")
    for i in range(num_days):
        if i < WINDOW_SIZE:
            # Default to Identity (Self-connection only)
            adj = np.eye(num_coins)
        else:
            # Slice window
            window_data = log_ret_df.iloc[i-WINDOW_SIZE : i]
            
            # Compute Pearson Correlation (-1 to 1)
            corr_matrix = window_data.corr(method='pearson').values
            corr_matrix = np.nan_to_num(corr_matrix, 0)
            
            # --- WEIGHTED EDGE LOGIC ---
            # 1. Absolute value
            adj = np.abs(corr_matrix)
            
            # 2. Thresholding
            adj[adj < THRESHOLD] = 0
            
            # 3. Self-Loops
            np.fill_diagonal(adj, 1.0)
            
        adj_matrices.append(adj)
        
    # Convert to Tensor
    adj_tensor = torch.FloatTensor(np.array(adj_matrices))
    
    # 5. Save
    torch.save(adj_tensor, OUTPUT_PATH)
    print(f"[+] Saved Rolling Matrices to {OUTPUT_PATH}")
    print(f"    Final Tensor Shape: {adj_tensor.shape}")
    
    # 6. Visualize Last Snapshot
    try:
        visualize_heatmap(adj_matrices[-1], coin_names)
    except Exception as e:
        print(f"Skipping visualization (Headless environment?): {e}")

def visualize_heatmap(matrix, labels):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, 
                xticklabels=labels, 
                yticklabels=labels, 
                cmap="viridis", 
                vmin=0, vmax=1,
                annot=False)
    
    plt.title(f"Weighted Correlation Snapshot (Last {WINDOW_SIZE} Days)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"    Heatmap saved to {PLOT_PATH}")
    # plt.show() 

if __name__ == "__main__":
    build_rolling_graph()