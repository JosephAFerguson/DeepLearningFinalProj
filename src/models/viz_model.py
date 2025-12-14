import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# ==========================================
# 1. DEFINE THE MODEL (So script is standalone)
# ==========================================

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features) # Visualization will show this!

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
        h_prime = torch.bmm(attention, h)
        return F.elu(self.norm(h_prime))

class DeepBiLSTMGNN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim, num_lstm_layers=2, num_gcn_layers=2, dropout=0.3):
        super(DeepBiLSTMGNN, self).__init__()
        
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
            dropout=dropout,
            bidirectional=True 
        )
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, _ = x.shape
        x_flat = x.view(batch_size * seq_len, num_nodes, -1)
        adj_flat = adj.view(batch_size * seq_len, num_nodes, num_nodes)
        
        # Path A: GNN
        gnn_out = x_flat
        for layer in self.gat_layers:
            gnn_out = layer(gnn_out, adj_flat)
        
        # Path B: Residual
        res_out = self.residual_proj(x_flat)
        
        # Combine
        gate = torch.clamp(self.alpha_gate, 0, 1.0) 
        combined = res_out + (gate * gnn_out)
        
        # LSTM
        out = combined.view(batch_size, seq_len, num_nodes, -1)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, -1)
        _, (h_n, _) = self.lstm(out)
        
        # Bi-Directional Concat
        final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        prediction = self.fc(final_state)
        return prediction.view(batch_size, num_nodes, 1)

# ==========================================
# 2. GENERATE DIAGRAM
# ==========================================

def generate_diagram():
    print("Generating High-Quality Architecture Diagram...")
    
    # --- Config for Visualization ---
    # Keep nodes/window small for the diagram so the graph isn't too cluttered
    VIZ_NODES = 5    # Show only 5 coins
    VIZ_WINDOW = 10  # Show only 10 days context
    VIZ_FEATS = 6    # Price, Vol, etc.
    HIDDEN_DIM = 32
    
    # 1. Initialize Model
    model = DeepBiLSTMGNN(num_nodes=VIZ_NODES, in_channels=VIZ_FEATS, hidden_dim=HIDDEN_DIM)
    
    # 2. Create Dummy Data
    x = torch.randn(1, VIZ_WINDOW, VIZ_NODES, VIZ_FEATS)
    adj = torch.randn(1, VIZ_WINDOW, VIZ_NODES, VIZ_NODES)
    
    # 3. Forward Pass (Capture the graph)
    y = model(x, adj)
    
    # 4. Create Dot Object
    # show_attrs=True -> Shows the shape of tensors at every step (Crucial for posters)
    # show_saved=True -> Shows where the gradients are saved
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # 5. Styling for "High Quality"
    # LR = Left-to-Right (looks more like a timeline/pipeline)
    # TB = Top-to-Bottom (Standard deep learning look)
    dot.attr(rankdir='TB') 
    
    # DPI and Font settings
    dot.attr(dpi='300')
    dot.attr(fontname='Helvetica')
    
    # 6. Render
    # We output PDF (Vector) for the poster, and PNG for quick viewing
    filename = "bilstm_gnn_architecture"
    
    # Render PDF
    dot.format = 'pdf'
    dot.render(filename)
    print(f"Saved vector graphic: {filename}.pdf")
    
    # Render PNG
    dot.format = 'png'
    dot.render(filename)
    print(f"Saved raster image: {filename}.png")

if __name__ == "__main__":
    generate_diagram()