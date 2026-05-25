import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Standard Graph Convolutional Network (GCN) layer.
    Computes H^(l+1) = sigma( D^{-0.5} A D^{-0.5} H^{(l)} W )
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, text, adj):
        # text: (Batch, N_nodes, in_features)
        # adj: (N_nodes, N_nodes) or (Batch, N_nodes, N_nodes)
        
        support = torch.matmul(text, self.weight)
        
        if len(adj.shape) == 2:
            output = torch.matmul(adj, support)
        else:
            output = torch.bmm(adj, support)
            
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNModel, self).__init__()
        self.gc1 = GraphConvLayer(in_dim, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, adj):
        # x: (Batch, N_nodes, Features)
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x
