import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ==========================================
# 1. Early Stopping Mechanism
# ==========================================
class EarlyStopping:
    """
    Stops training when validation loss stops improving after a set patience horizon.
    Saves the optimal model checkpoint weights.
    """
    def __init__(self, patience=10, min_delta=1e-4, checkpoint_path="best_stgnn_checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f" [+] Validation loss decreased. Checkpoint saved to: {self.checkpoint_path}")
        else:
            self.counter += 1
            print(f" [!] EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ==========================================
# 2. Production Node-Agnostic ST-GNN
# ==========================================
class ProductionSTGNN(nn.Module):
    """
    Production-grade Node-Agnostic ST-GNN with:
    - Vectorized Spatial Convolutions using GATv2Conv (No loops over time)
    - Dynamic Multi-Dimensional Edge Attribute Flattening
    - Vectorized Price Reconstruction to support non-stationary variables
    """
    def __init__(self, in_features, spatial_dim, temporal_dim, edge_features=1, lag_1_idx=0):
        super(ProductionSTGNN, self).__init__()
        self.lag_1_idx = lag_1_idx
        
        # GATv2 natively supports multidimensional edge attributes [E*T, F_edge]
        self.conv1 = GATv2Conv(in_features, spatial_dim, edge_dim=edge_features, heads=1, concat=False)
        self.conv2 = GATv2Conv(spatial_dim, spatial_dim, edge_dim=edge_features, heads=1, concat=False)
        
        self.norm1 = nn.LayerNorm(spatial_dim)
        self.norm2 = nn.LayerNorm(spatial_dim)
        
        # Shared Temporal GRU (processes each node sequence independently)
        self.shared_gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True
        )
        self.node_forecast_head = nn.Linear(temporal_dim, 1)

    def _vectorize_graph_temporal(self, x, edge_index, edge_attr, T):
        """
        Transforms spatial variables into a batch-temporal unified space.
        Decoupled from node counts and batch size.
        """
        B_N, _, F_in = x.shape
        x_flat = x.view(B_N * T, F_in)
        
        # Replicate and offset edge indices
        num_edges = edge_index.size(1)
        edge_index_flat = edge_index.repeat(1, T)
        
        offsets = torch.arange(T, device=x.device) * B_N
        offsets_repeated = torch.repeat_interleave(offsets, num_edges).unsqueeze(0)
        edge_index_flat = edge_index_flat + offsets_repeated
        
        # Replicate and flatten dynamic edge attributes
        edge_attr_flat = None
        if edge_attr is not None:
            if len(edge_attr.shape) == 2:
                edge_attr = edge_attr.unsqueeze(-1)
            # Permute to align with edge_index_flat: [E_spatial, T, F_edge] -> [T, E_spatial, F_edge]
            edge_attr_flat = edge_attr.permute(1, 0, 2).reshape(num_edges * T, -1)
            
        return x_flat, edge_index_flat, edge_attr_flat

    def forward(self, x, edge_index, edge_attr=None, batch_mapping=None):
        # x shape: [B_N, T, In_Features] where B_N = Batch * Nodes
        B_N, T, F_in = x.shape
        
        # 1. Flatten the temporal graph structure vectorially
        x_flat, edge_index_flat, edge_attr_flat = self._vectorize_graph_temporal(
            x, edge_index, edge_attr, T
        )
        
        # 2. Loop-free GATv2 Forward pass
        h1 = F.relu(self.conv1(x_flat, edge_index_flat, edge_attr=edge_attr_flat))
        h1 = self.norm1(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index_flat, edge_attr=edge_attr_flat))
        h_spatial_flat = self.norm2(h2)
        
        # 3. Reshape back to temporal sequence
        h_spatial = h_spatial_flat.view(B_N, T, -1)
        
        # 4. Shared Temporal Recurrence
        gru_out, _ = self.shared_gru(h_spatial)
        last_temporal_state = gru_out[:, -1, :] # [B_N, Temporal_Dim]
        
        # 5. Output predicted differences (Stationary Target)
        predicted_diffs = self.node_forecast_head(last_temporal_state) # [B_N, 1]
        
        # 6. Vectorized Price Reconstruction: y_t = y_{t-1} + delta_y_t
        y_last = x[:, -1, self.lag_1_idx].unsqueeze(-1)
        reconstructed_prices = y_last + predicted_diffs
        
        return reconstructed_prices, gru_out


# ==========================================
# 3. Velocity-Based MMD Domain Alignment
# ==========================================
class VelocityMMDLoss(nn.Module):
    """
    Computes MMD over stationary hidden state changes (velocities)
    to protect variance characteristics during domain transfer.
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(VelocityMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source_seq, target_seq):
        # Calculate stationary velocity vectors (First differences)
        source_vel = source_seq[:, 1:, :] - source_seq[:, :-1, :]
        target_vel = target_seq[:, 1:, :] - target_seq[:, :-1, :]
        
        # Flatten to Batch*Time space
        source_flat = source_vel.reshape(-1, source_vel.size(-1))
        target_flat = target_vel.reshape(-1, target_vel.size(-1))
        
        batch_size = min(source_flat.size(0), target_flat.size(0))
        s_sample = source_flat[:batch_size]
        t_sample = target_flat[:batch_size]
        
        kernels = self.gaussian_kernel(s_sample, t_sample, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)


# ==========================================
# 4. Domain Adaptation Training Pipeline
# ==========================================
def train_domain_adaptation_pipeline(
    model, source_loader, target_train_loader, target_val_loader,
    epochs=50, base_lr=1e-4, mmd_lambda=0.5, weight_decay=1e-3, patience=10, device="cpu"
):
    model.to(device)
    task_criterion = nn.MSELoss()
    mmd_criterion = VelocityMMDLoss()
    early_stopper = EarlyStopping(patience=patience, checkpoint_path="best_target_stgnn.pt")

    # Layer-Specific Discriminative Learning Rates
    optimizer_groups = [
        {"params": [p for n, p in model.named_parameters() if "conv" in n], "lr": base_lr * 0.1},
        {"params": [p for n, p in model.named_parameters() if "shared_gru" in n], "lr": base_lr},
        {"params": [p for n, p in model.named_parameters() if "node_forecast_head" in n], "lr": base_lr * 10.0}
    ]
    
    optimizer = AdamW(optimizer_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"\n🚀 Starting Production Fine-Tuning Loop on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_task_loss, train_mmd_loss, train_total_loss = 0.0, 0.0, 0.0
        source_iter = iter(source_loader)
        
        for target_batch in target_train_loader:
            target_batch = target_batch.to(device)
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)
            source_batch = source_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            _, source_gru_out = model(source_batch.x, source_batch.edge_index, source_batch.edge_attr)
            target_preds, target_gru_out = model(target_batch.x, target_batch.edge_index, target_batch.edge_attr)
            
            # Compute Losses
            task_loss = task_criterion(target_preds, target_batch.y)
            mmd_loss = mmd_criterion(source_gru_out, target_gru_out)
            total_loss = task_loss + (mmd_lambda * mmd_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_task_loss += task_loss.item() * target_batch.num_graphs
            train_mmd_loss += mmd_loss.item() * target_batch.num_graphs
            train_total_loss += total_loss.item() * target_batch.num_graphs

        num_train_samples = len(target_train_loader.dataset)
        avg_train_total = train_total_loss / num_train_samples

        # Validation Pass
        model.eval()
        val_task_loss = 0.0
        with torch.no_grad():
            for val_batch in target_val_loader:
                val_batch = val_batch.to(device)
                val_preds, _ = model(val_batch.x, val_batch.edge_index, val_batch.edge_attr)
                val_task_loss += task_criterion(val_preds, val_batch.y).item() * val_batch.num_graphs
                
        avg_val_loss = val_task_loss / len(target_val_loader.dataset)
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_total:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}.")
            break
            
    if os.path.exists(early_stopper.checkpoint_path):
        model.load_state_dict(torch.load(early_stopper.checkpoint_path))
    return model


# ==========================================
# 5. Automated Verification Execution Block
# ==========================================
if __name__ == "__main__":
    print("--- Running Automated Verification for transfer_learning.py ---")
    
    # 1. Structural Parameters
    in_features, edge_features, seq_len = 4, 2, 12
    spatial_dim, temporal_dim = 16, 32
    
    # Initialize Pipeline Component
    model = ProductionSTGNN(
        in_features=in_features, 
        spatial_dim=spatial_dim, 
        temporal_dim=temporal_dim, 
        edge_features=edge_features, 
        lag_1_idx=0
    )
    
    # 2. Verification Case A: Construct Source Graph Mock Data (Large Graph: 50 nodes)
    src_nodes, src_edges = 50, 120
    source_sample = Data(
        x=torch.randn(src_nodes, seq_len, in_features),
        edge_index=torch.randint(0, src_nodes, (2, src_edges)),
        edge_attr=torch.rand(src_edges, seq_len, edge_features),
        y=torch.randn(src_nodes, 1)
    )
    
    # 3. Verification Case B: Construct Target Graph Mock Data (Small Graph: 15 nodes)
    tgt_nodes, tgt_edges = 15, 45
    target_sample_train = Data(
        x=torch.randn(tgt_nodes, seq_len, in_features),
        edge_index=torch.randint(0, tgt_nodes, (2, tgt_edges)),
        edge_attr=torch.rand(tgt_edges, seq_len, edge_features),
        y=torch.randn(tgt_nodes, 1)
    )
    target_sample_val = copy.deepcopy(target_sample_train)
    
    # Assemble into variable-dimension DataLoaders
    src_loader = DataLoader([source_sample], batch_size=1)
    tgt_train_loader = DataLoader([target_sample_train], batch_size=1)
    tgt_val_loader = DataLoader([target_sample_val], batch_size=1)
    
    # 4. Execution Sanity Test (Single Epoch Run)
    try:
        trained_model = train_domain_adaptation_pipeline(
            model=model,
            source_loader=src_loader,
            target_train_loader=tgt_train_loader,
            target_val_loader=tgt_val_loader,
            epochs=1,
            patience=1,
            device="cpu"
        )
        print("\n[✔] SUCCESS: New Transfer Learning module verified without structural exceptions.")
    except Exception as e:
        print(f"\n[❌] ERROR: Verification failed with exception:\n{str(e)}")
