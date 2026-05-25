import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from torch_geometric.nn import GATv2Conv
from src.utils import get_food_columns

# ==========================================
# 1. FiLM (Feature-wise Linear Modulation)
# ==========================================
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Scales and shifts spatial hidden states using global macroeconomic context.
    """
    def __init__(self, global_dim, feature_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(global_dim, feature_dim)
        self.beta = nn.Linear(global_dim, feature_dim)

    def forward(self, x, global_features):
        # x: [Batch, Timesteps, Nodes, Feature_Dim]
        # global_features: [Batch, Timesteps, Global_Dim]
        gf = global_features.unsqueeze(2) # Shape: [Batch, Timesteps, 1, Global_Dim]
        g = torch.sigmoid(self.gamma(gf)) # Shape: [Batch, Timesteps, 1, Feature_Dim]
        b = self.beta(gf)                 # Shape: [Batch, Timesteps, 1, Feature_Dim]
        return g * x + b


# ==========================================
# 2. Multi-Target STGNN (Vectorized GATv2)
# ==========================================
class MultiTargetSTGNN(nn.Module):
    """
    Vectorized Multi-Target ST-GNN.
    - Treats all 145 commodities as nodes in a shared spatial graph.
    - Captures cross-commodity relationships via GATv2 neighborhood message passing.
    - Integrates global macroeconomic indicators via FiLM context conditioning.
    - Shared temporal GRU models time dependencies independently per node.
    """
    def __init__(self, n_nodes, in_dim, gcn_hidden, gru_hidden, global_dim=3):
        super(MultiTargetSTGNN, self).__init__()
        self.n_nodes = n_nodes
        self.gcn1 = GATv2Conv(in_dim, gcn_hidden, edge_dim=1, heads=1, concat=False)
        self.gcn2 = GATv2Conv(gcn_hidden, gcn_hidden, edge_dim=1, heads=1, concat=False)
        
        self.norm1 = nn.LayerNorm(gcn_hidden)
        self.norm2 = nn.LayerNorm(gcn_hidden)
        
        self.film = FiLM(global_dim, gcn_hidden)
        self.shared_gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        self.forecast_head = nn.Linear(gru_hidden, 1)

    def forward(self, x, adj, global_macro):
        # x: [B_N, T, In_Dim] where B_N = Batch * Nodes
        # adj: can be a tuple (edge_index, edge_weight) or just edge_index
        # global_macro: [Batch, T, Global_Dim]
        if isinstance(adj, tuple):
            edge_index, edge_weight = adj
        else:
            edge_index = adj
            edge_weight = None

        B_N, T, F_in = x.shape
        N = self.n_nodes
        B = B_N // N
        
        # 1. Reshape and permute to [B, T, N, F_in]
        x_rich = x.view(B, N, T, F_in).permute(0, 2, 1, 3) # [B, T, N, F_in]
        
        # Flatten temporal graph structure vectorially
        x_flat = x_rich.reshape(B * T * N, F_in)
        
        # 2. Replicate and offset edge indices
        num_edges = edge_index.size(1)
        edge_index_flat = edge_index.repeat(1, B * T)
        
        offsets = torch.arange(B * T, device=x.device) * N
        offsets_repeated = torch.repeat_interleave(offsets, num_edges).unsqueeze(0)
        edge_index_flat = edge_index_flat + offsets_repeated
        
        # 3. Replicate edge weights
        if edge_weight is not None:
            edge_weight_flat = edge_weight.repeat(B * T, 1)
        else:
            edge_weight_flat = None
        
        # 4. GATv2 Message Passing
        h1 = F.relu(self.conv1_forward(x_flat, edge_index_flat, edge_weight_flat))
        h1 = self.norm1(h1)
        h2 = F.relu(self.conv2_forward(h1, edge_index_flat, edge_weight_flat))
        h_spatial_flat = self.norm2(h2)
        
        # 5. Reshape back to sequence: [Batch, Timesteps, Nodes, GCN_Hidden]
        h_spatial = h_spatial_flat.view(B, T, N, -1) # [Batch, T, Nodes, GCN_Hidden]
        
        # Apply global macroeconomic context modulation
        h_modulated = self.film(h_spatial, global_macro)
        
        # 6. Prepare for shared GRU: [Batch * Nodes, T, GCN_Hidden]
        h_for_gru = h_modulated.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        gru_out, _ = self.shared_gru(h_for_gru)
        last_temporal_state = gru_out[:, -1, :] # [Batch * Nodes, GRU_Hidden]
        
        # 7. Output predicted returns
        preds_flat = self.forecast_head(last_temporal_state) # [Batch * Nodes, 1]
        
        # Reshape to [Batch, Nodes]
        return preds_flat.view(B, N)

    def conv1_forward(self, x, edge_index, edge_weight=None):
        return self.gcn1(x, edge_index, edge_attr=edge_weight)

    def conv2_forward(self, x, edge_index, edge_weight=None):
        return self.gcn2(x, edge_index, edge_attr=edge_weight)



# ==========================================
# 3. STGNNRegressor Pipeline Wrapper
# ==========================================
class STGNNRegressor(BaseEstimator, RegressorMixin):
    """
    Pipeline-compatible wrapper for training and evaluating the Multi-Target ST-GNN.
    - Caches the trained shared multi-commodity network to prevent redundant fits.
    - Reconstructs prices from stationary price returns dynamically.
    - Resolves recursive forecasting steps on single rows.
    """
    def __init__(self, epochs=200, batch_size=32, lr=5e-4, patience=30, seq_len=12, device=None, df=None, target=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.df = df
        self.target = target
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.history_returns = None
        
        self.shared_model_path = Path("models/saved_models/shared_multi_target_stgnn.pkl")
        self.feature_names = []
        self.nodes = []
        self.scalers = {}
        self.edge_index = None
        self.edge_weight = None

    def _prepare_returns_matrix(self, df_raw):
        """Prepares the stationary return inputs for all nodes."""
        if not self.nodes:
            food_cols = get_food_columns(df_raw)
            macro_cols = [c for c in ['Petrol (92 Octane)', 'USD_LKR', 'Food Index', 'Inflation'] if c in df_raw.columns]
            self.nodes = food_cols + macro_cols
        
        prices = df_raw[self.nodes].ffill().bfill().values
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)
        return returns

    def _prepare_global_macro(self, df_raw):
        """Constructs a 3-dimensional global macro state at each timestep."""
        petrol_col = 'Petrol (92 Octane)' if 'Petrol (92 Octane)' in df_raw.columns else df_raw.columns[0]
        usd_col = 'USD_LKR' if 'USD_LKR' in df_raw.columns else df_raw.columns[0]
        inf_col = 'Inflation' if 'Inflation' in df_raw.columns else df_raw.columns[0]
        
        p = df_raw[petrol_col].ffill().bfill().values
        u = df_raw[usd_col].ffill().bfill().values
        inf = df_raw[inf_col].ffill().bfill().values
        
        p_ret = np.zeros_like(p)
        p_ret[1:] = (p[1:] - p[:-1]) / (p[:-1] + 1e-8)
        
        u_ret = np.zeros_like(u)
        u_ret[1:] = (u[1:] - u[:-1]) / (u[:-1] + 1e-8)
        
        return np.column_stack([p_ret, u_ret, inf])

    def fit(self, X, y):
        # Save feature names list to locate `lag_1` index during prediction
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Check if pre-trained shared model exists
        if self.shared_model_path.exists():
            checkpoint = torch.load(self.shared_model_path, map_location=self.device, weights_only=False)
            checkpoint_nodes = checkpoint.get('nodes', [])
            
            # Check compatibility with current dataset features
            current_food_cols = get_food_columns(self.df)
            current_macro_cols = [c for c in ['Petrol (92 Octane)', 'USD_LKR', 'Food Index', 'Inflation'] if c in self.df.columns]
            current_nodes = current_food_cols + current_macro_cols
            
            if checkpoint_nodes == current_nodes:
                print(f" [+] Found pre-trained Shared Multi-Target ST-GNN. Loading checkpoint...")
                self.nodes = checkpoint_nodes
                self.scalers = checkpoint['scalers']
                
                if 'edge_index' in checkpoint:
                    self.edge_index = checkpoint['edge_index'].to(self.device)
                    self.edge_weight = checkpoint['edge_weight'].to(self.device)
                else:
                    adj_matrix = checkpoint['adj_matrix'].to(self.device)
                    self.edge_index = adj_matrix.nonzero().t().contiguous()
                    self.edge_weight = adj_matrix[self.edge_index[0], self.edge_index[1]].unsqueeze(1).contiguous()
                
                n_nodes = len(self.nodes)
                self.model = MultiTargetSTGNN(
                    n_nodes=n_nodes,
                    in_dim=1,
                    gcn_hidden=16,
                    gru_hidden=64,
                    global_dim=3
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Setup history returns for prediction
                returns = self._prepare_returns_matrix(self.df)
                split_idx = int(len(self.df) * 0.8)
                train_returns = returns[:split_idx]
                train_returns_scaled = self.scalers['x'].transform(train_returns)
                self.history_returns = train_returns_scaled[-self.seq_len:]
                
                return self
            else:
                print(f" [!] Incompatible cached ST-GNN found (node structure mismatch). Discarding cache and re-training...")
            
        print(f" [!] Shared model not found. Training Multi-Target ST-GNN...")
        # 1. Prepare Returns Matrix and Nodes
        returns = self._prepare_returns_matrix(self.df)
        n_nodes = len(self.nodes)
        
        # 2. Build Adjacency Matrix and convert to sparse representation
        from src.stgnn.adjacency import EconomicGraphBuilder
        graph_builder = EconomicGraphBuilder(self.df)
        G = graph_builder.build_correlation_graph(threshold=0.95)
        adj_matrix_np = graph_builder.get_adjacency_matrix(G)
        adj_matrix_t = torch.FloatTensor(adj_matrix_np)
        
        self.edge_index = adj_matrix_t.nonzero().t().contiguous().to(self.device)
        self.edge_weight = adj_matrix_t[self.edge_index[0], self.edge_index[1]].unsqueeze(1).contiguous().to(self.device)
        
        # 3. Train/Test Split (80/20 chronological)
        split_idx = int(len(self.df) * 0.8)
        train_returns = returns[:split_idx]
        
        # Fit standard scaler over returns
        scaler_x = StandardScaler()
        train_returns_scaled = scaler_x.fit_transform(train_returns)
        self.scalers['x'] = scaler_x
        
        # 4. Create sequences of shape [Samples, seq_len, Nodes, 1]
        X_seq, y_seq = [], []
        for i in range(len(train_returns_scaled) - self.seq_len):
            X_seq.append(train_returns_scaled[i : i + self.seq_len])
            y_seq.append(train_returns_scaled[i + self.seq_len])
            
        X_seq = np.array(X_seq)[..., np.newaxis] # [Samples, seq_len, Nodes, 1]
        y_seq = np.array(y_seq) # [Samples, Nodes]
        
        # 5. Global Macro features (USD_LKR return, Petrol return, Inflation return)
        global_macro = self._prepare_global_macro(self.df)[:split_idx]
        scaler_global = StandardScaler()
        global_macro_scaled = scaler_global.fit_transform(global_macro)
        self.scalers['global'] = scaler_global
        
        # Create global sequences: [Samples, seq_len, Global_Dim]
        X_global = []
        for i in range(len(global_macro_scaled) - self.seq_len):
            X_global.append(global_macro_scaled[i : i + self.seq_len])
        X_global = np.array(X_global)
        
        # Train dataset loaders
        X_t = torch.FloatTensor(X_seq).to(self.device)
        y_t = torch.FloatTensor(y_seq).to(self.device)
        global_t = torch.FloatTensor(X_global).to(self.device)
        
        dataset = TensorDataset(X_t, y_t, global_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 6. Instantiate model
        self.model = MultiTargetSTGNN(
            n_nodes=n_nodes,
            in_dim=1,
            gcn_hidden=16,
            gru_hidden=64,
            global_dim=3
        ).to(self.device)
        
        # Zero initialize head to start with persistence baseline
        nn.init.zeros_(self.model.forecast_head.weight)
        nn.init.zeros_(self.model.forecast_head.bias)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for bx, by, bg in loader:
                optimizer.zero_grad()
                B, T, N, F_in = bx.shape
                bx_flat = bx.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
                
                preds = self.model(bx_flat, (self.edge_index, self.edge_weight), bg)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
            scheduler.step()
            
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
            
        # Cache history returns
        self.history_returns = train_returns_scaled[-self.seq_len:]
        
        # Save model and components to disk
        self.shared_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'nodes': self.nodes,
            'edge_index': self.edge_index.cpu(),
            'edge_weight': self.edge_weight.cpu(),
            'scalers': self.scalers
        }, self.shared_model_path)
        
        return self

    def predict(self, X):
        self.model.eval()
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        N_samples = len(X_vals)
        
        target_idx = self.nodes.index(self.target)
        
        # Determine lag_1 index
        if 'lag_1' in self.feature_names:
            lag_1_idx = self.feature_names.index('lag_1')
        else:
            lag_1_idx = 0
            
        y_last_vals = X_vals[:, lag_1_idx]
        
        # Scenario A: Step-by-Step Recursive Forecasting (len(X) == 1)
        if N_samples == 1:
            bx = torch.FloatTensor(self.history_returns[np.newaxis, ..., np.newaxis]).to(self.device)
            B, T, N, F_in = bx.shape
            bx_flat = bx.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
            
            global_raw = self._prepare_global_macro(self.df)
            global_scaled = self.scalers['global'].transform(global_raw)
            
            if hasattr(X, 'index') and len(X.index) > 0:
                dt = X.index[0]
                idx = self.df.index.get_loc(dt)
                bg_seq = global_scaled[max(0, idx - self.seq_len + 1) : idx + 1]
                if len(bg_seq) < self.seq_len:
                    pad = np.repeat(global_scaled[0:1], self.seq_len - len(bg_seq), axis=0)
                    bg_seq = np.vstack([pad, bg_seq])
            else:
                bg_seq = np.zeros((self.seq_len, 3))
                
            bg = torch.FloatTensor(bg_seq[np.newaxis]).to(self.device)
            
            with torch.no_grad():
                preds_scaled = self.model(bx_flat, (self.edge_index, self.edge_weight), bg).cpu().numpy()
                
            preds_unscaled = self.scalers['x'].inverse_transform(preds_scaled).flatten()
            pred_return = preds_unscaled[target_idx]
            
            y_pred = y_last_vals[0] * (1.0 + pred_return)
            
            # Update history returns
            next_step_returns = preds_scaled[0]
            self.history_returns = np.vstack([self.history_returns[1:], next_step_returns])
            
            return np.array([y_pred])
            
        # Scenario B: Test Set Evaluation (len(X) > 1)
        else:
            returns = self._prepare_returns_matrix(self.df)
            returns_scaled = self.scalers['x'].transform(returns)
            
            test_indices = []
            for dt in X.index:
                test_indices.append(self.df.index.get_loc(dt))
                
            global_raw = self._prepare_global_macro(self.df)
            global_scaled = self.scalers['global'].transform(global_raw)
            
            predictions = []
            for idx in test_indices:
                seq_start = idx - self.seq_len
                if seq_start < 0:
                    pad = np.zeros((-seq_start, len(self.nodes)))
                    seq = np.vstack([pad, returns_scaled[:idx]])
                else:
                    seq = returns_scaled[seq_start:idx]
                    
                bx = torch.FloatTensor(seq[np.newaxis, ..., np.newaxis]).to(self.device)
                B, T, N, F_in = bx.shape
                bx_flat = bx.permute(0, 2, 1, 3).reshape(B * N, T, F_in)
                
                if seq_start < 0:
                    pad = np.zeros((-seq_start, 3))
                    bg_seq = np.vstack([pad, global_scaled[:idx]])
                else:
                    bg_seq = global_scaled[seq_start:idx]
                bg = torch.FloatTensor(bg_seq[np.newaxis]).to(self.device)
                
                with torch.no_grad():
                    preds_scaled = self.model(bx_flat, (self.edge_index, self.edge_weight), bg).cpu().numpy()
                    
                preds_unscaled = self.scalers['x'].inverse_transform(preds_scaled).flatten()
                pred_return = preds_unscaled[target_idx]
                predictions.append(pred_return)
                
            predictions = np.array(predictions)
            y_preds = y_last_vals * (1.0 + predictions)
            return y_preds
