import torch
import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32, num_layers=2, dropout=0.3):
        super(VanillaLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Standard LSTM
        # We treat every coin's history as an independent sequence.
        self.lstm = nn.LSTM(
            input_size=in_channels, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        # Prediction Head
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj=None):
        """
        x: (Batch, Window, Nodes, Features)
        adj: Ignored (The baseline doesn't care about neighbors!)
        """
        batch_size, window_size, num_nodes, features = x.shape
        
        # 1. Reshape to treat (Batch * Nodes) as independent samples
        # effectively stacking all coins' histories into one giant batch
        x_flat = x.permute(0, 2, 1, 3).contiguous() # (Batch, Nodes, Window, Feats)
        x_flat = x_flat.view(batch_size * num_nodes, window_size, features)
        
        # 2. Run LSTM
        # out: (Batch*Nodes, Window, Hidden)
        # h_n: (Num_Layers, Batch*Nodes, Hidden)
        lstm_out, (h_n, c_n) = self.lstm(x_flat)
        
        # Take the last hidden state
        final_state = h_n[-1] 
        
        # 3. Predict
        prediction = self.fc(final_state) # (Batch*Nodes, 1)
        
        # 4. Reshape back to original format (Batch, Nodes, 1)
        return prediction.view(batch_size, num_nodes, 1)