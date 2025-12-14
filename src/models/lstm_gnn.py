import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. New GAT Layer (The "Silver Bullet") ---
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        # NEW: Layer Norm to prevent explosion
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        h = self.W(x) 
        batch_size, num_nodes, _ = h.size()
        
        h_i = h.unsqueeze(2).repeat(1, 1, num_nodes, 1) 
        h_j = h.unsqueeze(1).repeat(1, num_nodes, 1, 1) 
        
        a_input = torch.cat([h_i, h_j], dim=-1) 
        e = self.leakyrelu(self.a(a_input).squeeze(-1)) 
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.bmm(attention, h)
        
        # Apply Norm and Activation
        return F.elu(self.norm(h_prime))

class DeepLSTMGNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim, num_lstm_layers=2, num_gcn_layers=2, dropout=0.3):
        super(DeepLSTMGNN, self).__init__()
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            input_dim = in_channels if i == 0 else hidden_dim
            self.gat_layers.append(GATLayer(input_dim, hidden_dim, dropout=dropout))
            
        self.residual_proj = nn.Linear(in_channels, hidden_dim)
        self.alpha_gate = nn.Parameter(torch.tensor(0.01))

        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_lstm_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, _ = x.shape
        x_flat = x.view(batch_size * seq_len, num_nodes, -1)
        adj_flat = adj.view(batch_size * seq_len, num_nodes, num_nodes)
        
        # --- PATH A: Run GNN ---
        gnn_out = x_flat
        for layer in self.gat_layers:
            gnn_out = layer(gnn_out, adj_flat)
        
        # --- PATH B: Residual ---
        res_out = self.residual_proj(x_flat)
        
        # --- COMBINE ---
        # SAFETY CLAMP: Ensure alpha_gate doesn't get too huge
        gate = torch.clamp(self.alpha_gate, 0, 1.0) 
        combined = res_out + (gate * gnn_out)
        
        out = combined.view(batch_size, seq_len, num_nodes, -1)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, -1)
        
        _, (h_n, _) = self.lstm(out)
        prediction = self.fc(h_n[-1])
        
        return prediction.view(batch_size, num_nodes, 1)

# Joe's code + alpha gate
class DeepBiLSTMGNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim, num_lstm_layers=2, num_gcn_layers=2, dropout=0.3):
        super(DeepBiLSTMGNN, self).__init__()
        
        # --- 1. Graph Layers (GAT) ---
        self.gat_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            input_dim = in_channels if i == 0 else hidden_dim
            # Ensure you are using the GATLayer class with LayerNorm we defined earlier!
            self.gat_layers.append(GATLayer(input_dim, hidden_dim, dropout=dropout))
            
        # --- 2. Residual Projection ---
        self.residual_proj = nn.Linear(in_channels, hidden_dim)
        self.alpha_gate = nn.Parameter(torch.tensor(0.01))

        # --- 3. Bi-LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True # Enabled
        )

        # Output layer must be 2x hidden_dim because of bidirectionality
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        x_flat = x.view(batch_size * seq_len, num_nodes, -1)
        adj_flat = adj.view(batch_size * seq_len, num_nodes, num_nodes)
        
        # --- PATH A: GNN ---
        gnn_out = x_flat
        for layer in self.gat_layers:
            gnn_out = layer(gnn_out, adj_flat)
        
        # --- PATH B: Residual ---
        res_out = self.residual_proj(x_flat)
        
        # --- COMBINE (With Safety Clamp) ---
        # Clamp gate between 0 and 1 to prevent explosion
        gate = torch.clamp(self.alpha_gate, 0, 1.0) 
        combined = res_out + (gate * gnn_out)
        
        # Reshape for LSTM
        out = combined.view(batch_size, seq_len, num_nodes, -1)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, -1)
        
        # Run Bi-LSTM
        _, (h_n, _) = self.lstm(out)
        
        # Concatenate the final forward and backward states
        h_n_forward = h_n[-2]
        h_n_backward = h_n[-1]
        final_state = torch.cat([h_n_forward, h_n_backward], dim=1)

        prediction = self.fc(final_state)
                
        return prediction.view(batch_size, num_nodes, 1)