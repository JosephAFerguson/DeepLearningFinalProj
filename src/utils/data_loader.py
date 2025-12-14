import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

class CryptoGraphDataset(Dataset):
    def __init__(self, prices_path, global_path, adj_path, window_size=60, prediction_step=1, train_ratio=0.8):
        self.window_size = window_size
        self.prediction_step = prediction_step
        
        # --- 1. Load Crypto Data (Micro) ---
        df = pd.read_csv(prices_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # --- FIX: REMOVE DUPLICATES ---
        # Ensure we don't have two rows for the same coin on the same day
        df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
        
        df = df.sort_values(['coin_id', 'date'])
        
        # Pivot everything to (Date x Coin)
        # We use 'ffill' to handle weekends/holidays if any data is missing
        price_df = df.pivot(index='date', columns='symbol', values='price').ffill().bfill()
        vol_df = df.pivot(index='date', columns='symbol', values='volume').ffill().bfill().fillna(0)
        
        self.coin_names = price_df.columns.tolist()
        dates = price_df.index
        
        # --- 2. Feature Engineering (Derived) ---
        
        # A. Log Returns (Stabilizes the trend)
        # ln(Pt / Pt-1)
        eps = 1e-8 # Small epsilon to avoid log(0)
        log_ret_df = np.log(price_df + eps) - np.log(price_df.shift(1) + eps)
        log_ret_df = log_ret_df.fillna(0)
        
        # B. Volatility (30-Day Rolling Std Dev)
        volatility_df = log_ret_df.rolling(window=30).std().fillna(0)
        
        # --- 3. Load Global Metrics (Macro) ---
        global_df = pd.read_csv(global_path)
        global_df['date'] = pd.to_datetime(global_df['date'])
        
        # Fix duplicates in Global Metrics too (just in case)
        global_df = global_df.drop_duplicates(subset=['date'], keep='last')
        global_df = global_df.set_index('date').sort_index()
        
        # Reindex global data to match our exact price dates
        global_df = global_df.reindex(dates).ffill().bfill()
        
        # Create full matrices for Macro features
        num_coins = len(self.coin_names)
        
        # Shape: (Num_Days, Num_Coins)
        btc_dom_matrix = np.tile(global_df['btc_dominance'].values[:, None], (1, num_coins))
        mkt_cap_matrix = np.tile(global_df['total_market_cap'].values[:, None], (1, num_coins))

        # --- 4. Scaling (Strict Train/Test Split) ---
        raw_data_list = [
            price_df.values,        # Ch 0
            vol_df.values,          # Ch 1
            log_ret_df.values,      # Ch 2
            volatility_df.values,   # Ch 3
            btc_dom_matrix,         # Ch 4
            mkt_cap_matrix          # Ch 5
        ]
        
        split_idx = int(len(dates) * train_ratio)
        self.scalers = []
        scaled_data_list = []
        
        for raw in raw_data_list:
            scaler = RobustScaler()
            scaler.fit(raw[:split_idx]) # Fit on TRAIN only
            scaled = scaler.transform(raw)
            scaled_data_list.append(scaled)
            self.scalers.append(scaler)
            
        # Stack into final tensor: (Days, Coins, 6)
        self.data = np.dstack(scaled_data_list)
        
        # --- 5. Load Graph ---
        self.adj_matrices = torch.load(adj_path)
        
        # Validation: Clip to shorter length
        min_len = min(len(self.data), len(self.adj_matrices))
        self.data = self.data[:min_len]
        self.adj_matrices = self.adj_matrices[:min_len]

    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_step

    def __getitem__(self, idx):
        i = idx + self.window_size
        
        # X: (Window, Nodes, 6)
        x_val = self.data[i - self.window_size : i]
        x_tensor = torch.FloatTensor(x_val)
        
        # Adj: (Window, Nodes, Nodes)
        adj_val = self.adj_matrices[i - self.window_size : i]
        
        # Y: Predict Price (Channel 0)
        y_val = self.data[i + self.prediction_step - 1, :, 0] 
        y_tensor = torch.FloatTensor(y_val).unsqueeze(-1)
        
        return x_tensor, adj_val, y_tensor
    
    def inverse_transform(self, y_scaled):
        # Helper for Channel 0 (Price)
        return self.scalers[0].inverse_transform(y_scaled)