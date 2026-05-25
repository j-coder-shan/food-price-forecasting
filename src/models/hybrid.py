import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.gnn import STGNNRegressor, MultiTargetSTGNN
from src.feature_engineering import FeatureEngineer

class ResidualSTGNNRegressor(STGNNRegressor):
    """
    Subclass of STGNNRegressor customized to train directly on target residuals.
    Replaces target returns with standardized residuals, bypassing raw price returns.
    """
    def __init__(self, epochs=200, batch_size=32, lr=5e-4, patience=30, seq_len=12, device=None, df=None, target=None):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            seq_len=seq_len,
            device=device,
            df=df,
            target=target
        )
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        # Save feature names list to locate `lag_1` index during prediction
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        print(f" [!] Training Residual Multi-Target ST-GNN...")
        
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
        
        # 3. Train/Test Split (80% training split)
        split_idx = int(len(self.df) * 0.8)
        train_returns = returns[:split_idx]
        
        # Fit standard scaler over returns
        scaler_x = StandardScaler()
        train_returns_scaled = scaler_x.fit_transform(train_returns)
        self.scalers['x'] = scaler_x
        
        # 4. Standard scale target residuals `y` (Double-Scaling)
        y_vals = y.values if hasattr(y, 'values') else np.array(y)
        y_scaled = self.scaler_y.fit_transform(y_vals.reshape(-1, 1)).flatten()
        y_scaled_series = pd.Series(y_scaled, index=y.index if hasattr(y, 'index') else None)
        
        # Replace target commodity column in train_returns_scaled with scaled residuals
        target_idx = self.nodes.index(self.target)
        train_dates = self.df.index[:split_idx]
        
        for dt, val in y_scaled_series.items():
            if dt in train_dates:
                idx = train_dates.get_loc(dt)
                train_returns_scaled[idx, target_idx] = val
                
        # 5. Create sequences of shape [Samples, seq_len, Nodes, 1]
        X_seq, y_seq = [], []
        for i in range(len(train_returns_scaled) - self.seq_len):
            X_seq.append(train_returns_scaled[i : i + self.seq_len])
            y_seq.append(train_returns_scaled[i + self.seq_len])
            
        X_seq = np.array(X_seq)[..., np.newaxis]  # [Samples, seq_len, Nodes, 1]
        y_seq = np.array(y_seq)  # [Samples, Nodes]
        
        # 6. Global Macro features (USD_LKR return, Petrol return, Inflation return)
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
        
        # 7. Instantiate model
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
                
            # For the target node, the prediction is the scaled residual
            pred_residual_scaled = preds_scaled[0, target_idx]
            pred_residual = self.scaler_y.inverse_transform([[pred_residual_scaled]])[0, 0]
            
            # Update history returns
            next_step_returns = preds_scaled[0]
            self.history_returns = np.vstack([self.history_returns[1:], next_step_returns])
            
            return np.array([pred_residual])
            
        # Scenario B: Test Set Evaluation (len(X) > 1)
        else:
            returns = self._prepare_returns_matrix(self.df)
            returns_scaled = self.scalers['x'].transform(returns)
            
            # We must replace the target column in returns_scaled with the scaled residuals
            fe = FeatureEngineer(self.df, self.target)
            df_features = fe.build_features()
            feature_cols = [c for c in df_features.columns if c != self.target]
            X_all = df_features[feature_cols]
            y_all = df_features[self.target]
            
            # Predict linear component for the whole dataset to compute actual residuals
            y_linear_all = self.linear_model.predict(X_all)
            residuals_all = y_all - y_linear_all
            
            residuals_all_scaled = self.scaler_y.transform(residuals_all.values.reshape(-1, 1)).flatten()
            residuals_all_scaled_series = pd.Series(residuals_all_scaled, index=residuals_all.index)
            
            # Replace target column in returns_scaled
            target_idx = self.nodes.index(self.target)
            for dt, val in residuals_all_scaled_series.items():
                if dt in self.df.index:
                    idx = self.df.index.get_loc(dt)
                    returns_scaled[idx, target_idx] = val
            
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
                    
                pred_residual_scaled = preds_scaled[0, target_idx]
                pred_residual = self.scaler_y.inverse_transform([[pred_residual_scaled]])[0, 0]
                predictions.append(pred_residual)
                
            return np.array(predictions)


class STGNNLinearHybridRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for the Hybrid Linear Regression + STGNN model.
    Couples a linear baseline with spatio-temporal graph modeling via residual learning.
    """
    def __init__(self, epochs=200, batch_size=32, lr=5e-3, patience=15, device=None, seq_len=12, df=None, target=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.device = device
        self.seq_len = seq_len
        self.df = df
        self.target = target
        
        self.linear_model = LinearRegression()
        self.stgnn_model = ResidualSTGNNRegressor(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            seq_len=self.seq_len,
            device=self.device,
            df=self.df,
            target=self.target
        )

    def fit(self, X, y):
        # Fit the LinearRegression model on (X, y)
        self.linear_model.fit(X, y)
        
        # Generate the in-sample predictions from the linear model
        y_linear = self.linear_model.predict(X)
        
        # Calculate the residuals
        residuals = y - y_linear
        
        # Fit the internal STGNNRegressor using X as the features and the residuals as the target
        self.stgnn_model.linear_model = self.linear_model
        self.stgnn_model.fit(X, residuals)
        
        return self

    def predict(self, X):
        # Generate linear predictions
        preds_linear = self.linear_model.predict(X)
        
        # Generate non-linear structural residual predictions from the trained ST-GNN
        self.stgnn_model.linear_model = self.linear_model
        preds_residual = self.stgnn_model.predict(X)
        
        # Combine them
        preds_final = preds_linear + preds_residual
        
        # Return 1D array of forecasts
        return preds_final
