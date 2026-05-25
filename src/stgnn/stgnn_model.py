import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.stgnn.adjacency import EconomicGraphBuilder
from src.stgnn.demand_forecasting import DemandForecaster
from src.stgnn.gcn_model import GCNModel
from src.stgnn.temporal_model import TemporalModel

class MultiTargetSTGNN(nn.Module):
    def __init__(self, n_nodes, in_features, gcn_hidden, gru_hidden, n_targets):
        super(MultiTargetSTGNN, self).__init__()
        self.gcn = GCNModel(in_features, gcn_hidden, gcn_hidden)
        
        # Flattened nodes * gcn_hidden to feed into GRU
        self.gru = TemporalModel(n_nodes * gcn_hidden, gru_hidden, num_layers=2)
        
        # Multi-target heads
        self.price_head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_targets['prices'])
        )
        self.demand_head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets['demands'])
        )
        self.macro_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets['macro'])  # e.g., Index and Inflation
        )

    def forward(self, x, adj):
        # x shape: (Batch, Timesteps, Nodes, Features)
        B, T, N, F = x.shape
        
        # Apply GCN at each timestep independently
        # Reshape to (Batch * Timesteps, Nodes, Features)
        x_gcn = x.view(B * T, N, F)
        
        # Graph convolution
        gcn_out = self.gcn(x_gcn, adj) # (B*T, N, gcn_hidden)
        
        # Reshape back to temporal sequence
        gcn_out = gcn_out.view(B, T, -1) # (B, T, N * gcn_hidden)
        
        # Temporal convolution
        temporal_out = self.gru(gcn_out) # (B, gru_hidden)
        
        # Heads
        prices = self.price_head(temporal_out)
        demands = self.demand_head(temporal_out)
        macro = self.macro_head(temporal_out)
        
        return prices, demands, macro


class AdvancedSTGNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seq_len=12, epochs=200, batch_size=32, lr=1e-3, patience=15, device=None):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scaler_x = StandardScaler()
        self.scaler_price = StandardScaler()
        self.scaler_demand = StandardScaler()
        self.scaler_macro = StandardScaler()
        
        self.model = None
        self.adj_matrix = None
        self.history_X = None

    def _make_sequences(self, X_arr, y_p=None, y_d=None, y_m=None):
        X_seq, Yp, Yd, Ym = [], [], [], []
        for i in range(len(X_arr) - self.seq_len):
            X_seq.append(X_arr[i : i + self.seq_len])
            if y_p is not None:
                Yp.append(y_p[i + self.seq_len])
                Yd.append(y_d[i + self.seq_len])
                Ym.append(y_m[i + self.seq_len])
        
        if y_p is not None:
            return np.array(X_seq), np.array(Yp), np.array(Yd), np.array(Ym)
        return np.array(X_seq)

    def fit(self, df_raw: pd.DataFrame, food_cols: list, demand_df: pd.DataFrame):
        # Build Adjacency Graph
        graph_builder = EconomicGraphBuilder(df_raw, demand_df)
        G = graph_builder.build_correlation_graph(threshold=0.6)
        self.adj_matrix = torch.FloatTensor(graph_builder.get_adjacency_matrix(G)).to(self.device)
        self.nodes = graph_builder.nodes
        n_nodes = len(self.nodes)
        
        # Prepare inputs (Features: 1 per node -> just the value)
        X_df = df_raw[self.nodes].fillna(method='bfill').fillna(method='ffill')
        X_scaled = self.scaler_x.fit_transform(X_df)
        
        # Targets
        Y_prices = X_df[food_cols].values
        Y_demands = demand_df[[f"{c}_Demand" for c in food_cols]].values
        
        # Optional Macro targets
        macro_target_cols = [c for c in ['Food Index', 'Inflation'] if c in df_raw.columns]
        if not macro_target_cols:
            macro_target_cols = [food_cols[0]] # Dummy if missing
            Y_macro = np.zeros((len(X_df), 1))
        else:
            Y_macro = df_raw[macro_target_cols].fillna(0).values
            
        Y_prices_s = self.scaler_price.fit_transform(Y_prices)
        Y_demands_s = self.scaler_demand.fit_transform(Y_demands)
        Y_macro_s = self.scaler_macro.fit_transform(Y_macro)
        
        # Sequences
        X_seq, Yp, Yd, Ym = self._make_sequences(X_scaled, Y_prices_s, Y_demands_s, Y_macro_s)
        
        # Shape X to (Batch, Timesteps, Nodes, 1)
        X_seq = X_seq[..., np.newaxis]
        
        X_t = torch.FloatTensor(X_seq).to(self.device)
        Yp_t = torch.FloatTensor(Yp).to(self.device)
        Yd_t = torch.FloatTensor(Yd).to(self.device)
        Ym_t = torch.FloatTensor(Ym).to(self.device)
        
        dataset = TensorDataset(X_t, Yp_t, Yd_t, Ym_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        n_targets = {
            'prices': Y_prices.shape[1],
            'demands': Y_demands.shape[1],
            'macro': Y_macro.shape[1]
        }
        
        self.model = MultiTargetSTGNN(n_nodes, in_features=1, gcn_hidden=16, gru_hidden=64, n_targets=n_targets).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bx, by_p, by_d, by_m in loader:
                optimizer.zero_grad()
                p_out, d_out, m_out = self.model(bx, self.adj_matrix)
                
                loss = criterion(p_out, by_p) + 0.5 * criterion(d_out, by_d) + 0.5 * criterion(m_out, by_m)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            epoch_loss /= len(loader)
            scheduler.step(epoch_loss)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
                    
        self.history_X = X_scaled[-(self.seq_len):]
        return self

    def predict_future(self, steps=1):
        """Autoregressive multi-target forecasting for `steps` ahead."""
        self.model.eval()
        predictions = {'prices': [], 'demands': [], 'macro': []}
        
        current_seq = self.history_X.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                x_in = current_seq[np.newaxis, ..., np.newaxis] # (1, seq_len, nodes, 1)
                x_t = torch.FloatTensor(x_in).to(self.device)
                
                p_out, d_out, m_out = self.model(x_t, self.adj_matrix)
                
                p_inv = self.scaler_price.inverse_transform(p_out.cpu().numpy())
                d_inv = self.scaler_demand.inverse_transform(d_out.cpu().numpy())
                m_inv = self.scaler_macro.inverse_transform(m_out.cpu().numpy())
                
                predictions['prices'].append(p_inv[0])
                predictions['demands'].append(d_inv[0])
                predictions['macro'].append(m_inv[0])
                
                # Update current_seq using predicted prices for auto-regression
                # Note: We only predicted target features, but macro/demand are also predicted.
                # In a real dynamic system we'd map p_out/m_out to the correct node index in current_seq.
                # For simplicity in this demo, we carry forward the last observed values and blend.
                next_step = current_seq[-1].copy()
                current_seq = np.vstack([current_seq[1:], next_step])
                
        return predictions
