import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

class AdvancedGraphStructureLearner(nn.Module):
    """Learns dynamic spatial correlations between macroeconomic and lag nodes."""
    def __init__(self, n_nodes, embed_dim=16, top_k=8):
        super().__init__()
        self.node_emb = nn.Embedding(n_nodes, embed_dim)
        self.top_k = min(top_k, n_nodes - 1) if n_nodes > 1 else 0

    def forward(self):
        e = F.normalize(self.node_emb.weight, dim=-1)
        sim = torch.mm(e, e.T)
        
        if self.top_k > 0:
            mask = torch.zeros_like(sim)
            mask.scatter_(1, sim.topk(self.top_k + 1, dim=-1).indices, 1)
            mask.fill_diagonal_(0)
            A = sim * mask
        else:
            A = sim
        return A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)

class TemporalConvNet(nn.Module):
    """Dilated 1D convolutions with causal padding and Gated Linear Units (GLU)."""
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, 
                              padding=self.padding, dilation=dilation)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        # x: (Batch * Nodes, Features, Timesteps)
        out = self.conv(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        return self.glu(out)

class STGCNLayer(nn.Module):
    """Spatio-Temporal layer handling sequences over graphs."""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(in_dim, out_dim)
        self.gcn_weight = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A):
        # H shape: (Batch, Nodes, Timesteps, Feature_Dim)
        B, N, T, F_in = H.shape
        
        # Temporal Conv per node
        h_t = H.view(B * N, F_in, T)
        h_t = self.tcn(h_t) # (B*N, out_dim, T)
        h_t = h_t.view(B, N, -1, T).permute(0, 3, 1, 2) # (B, T, N, out_dim)
        
        # Expand Adjacency matrix across batch dimension
        A_expanded = A.unsqueeze(0).expand(B, -1, -1)
        
        # Graph convolution per timestep
        out = []
        for t in range(T):
            g_out = torch.bmm(A_expanded, h_t[:, t, :, :])
            out.append(self.gcn_weight(g_out).unsqueeze(1))
            
        out = torch.cat(out, dim=1) # (B, T, N, out_dim)
        out = self.norm(out)
        return F.relu(self.drop(out)).permute(0, 2, 1, 3) # (B, N, T, out_dim)

class AdvancedSTGNNExtractor(nn.Module):
    def __init__(self, n_nodes, seq_len=6, in_dim=1, out_dim=32):
        super().__init__()
        self.gsl = AdvancedGraphStructureLearner(n_nodes, top_k=min(8, n_nodes))
        self.stgcn = STGCNLayer(in_dim, out_dim)
        
        # Flatten structural nodes AND temporal dimension
        self.head = nn.Sequential(
            nn.Linear(n_nodes * seq_len * out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x input shape: (Batch, Nodes, Timesteps, Feature_Dim)
        A = self.gsl()
        H = self.stgcn(x, A) # Shape: (Batch, Nodes, Timesteps, out_dim)
        
        # Flatten representations
        emb = H.reshape(H.size(0), -1)
        return self.head(emb)

class CombinedRobustLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, y_pred, y_true):
        l1_loss = self.smooth_l1(y_pred, y_true)
        mask = y_true != 0
        if mask.sum() > 0:
            mape_loss = torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        else:
            mape_loss = 0.0
        return self.alpha * l1_loss + self.beta * mape_loss

class STGNNRegressor(BaseEstimator, RegressorMixin):
    """Clean drop-in replacement for your model evaluation framework."""
    def __init__(self, epochs=200, batch_size=32, lr=5e-3, patience=15, seq_len=6, device=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.history_X = None

    def _make_sequences(self, x_arr, y_arr=None):
        X_seq, y_seq = [], []
        for i in range(len(x_arr) - self.seq_len + 1):
            X_seq.append(x_arr[i : i + self.seq_len])
            if y_arr is not None:
                y_seq.append(y_arr[i + self.seq_len - 1])
        if y_arr is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq)

    def fit(self, X, y):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        
        X_scaled = self.scaler_x.fit_transform(X_vals)
        y_scaled = self.scaler_y.fit_transform(y_vals.reshape(-1, 1))
        
        X_seq, y_seq = self._make_sequences(X_scaled, y_scaled)
        
        if len(X_seq) == 0:
            raise ValueError(f"Dataset too small for seq_len {self.seq_len}")
            
        # Cache the history for predict()
        if len(X_scaled) >= self.seq_len - 1:
            self.history_X = X_scaled[-(self.seq_len - 1):]
        else:
            self.history_X = X_scaled
        
        B, T, N_features = X_seq.shape
        
        # Shape: (Batch, Nodes, Timesteps, Feature_Dim)
        X_t = torch.tensor(X_seq, dtype=torch.float32).permute(0, 2, 1).unsqueeze(3).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).to(self.device)

        self.model = AdvancedSTGNNExtractor(n_nodes=N_features, seq_len=self.seq_len, in_dim=1, out_dim=32).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        criterion = CombinedRobustLoss(alpha=1.0, beta=0.1)
        
        best_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
                    
        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
        return self

    def predict(self, X):
        self.model.eval()
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler_x.transform(X_vals)
        
        # Prepend cached history for continuous sequence forecasting
        if self.history_X is not None and len(self.history_X) > 0:
            combined_X = np.vstack([self.history_X, X_scaled])
        else:
            combined_X = X_scaled
            
        X_seq = self._make_sequences(combined_X)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough history. Expected at least {self.seq_len} rows, got {len(combined_X)}.")
            
        X_t = torch.tensor(X_seq, dtype=torch.float32).permute(0, 2, 1).unsqueeze(3).to(self.device)
        
        with torch.no_grad():
            preds_scaled = self.model(X_t).cpu().numpy()
            
        # Update history for recursive predictions (e.g. next step forecast)
        if len(combined_X) >= self.seq_len - 1:
            self.history_X = combined_X[-(self.seq_len - 1):]
            
        return self.scaler_y.inverse_transform(preds_scaled).flatten()
