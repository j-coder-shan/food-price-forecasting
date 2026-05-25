import numpy as np
import pandas as pd
import networkx as nx
from src.utils import get_food_columns

class EconomicGraphBuilder:
    """
    Constructs the Adjacency Matrix and NetworkX graph for the ST-GNN.
    Nodes: Food Items, Fuel, USD/LKR, Food Index, Inflation, Demand.
    Edges: Relationships based on correlation or domain knowledge.
    """
    def __init__(self, df: pd.DataFrame, demand_df: pd.DataFrame = None):
        self.df = df
        self.demand_df = demand_df
        self.food_cols = get_food_columns(self.df)
        self.macro_cols = ['Petrol (92 Octane)', 'USD_LKR', 'Food Index', 'Inflation']
        
        # All valid nodes present in the dataframe
        self.nodes = []
        for c in self.food_cols:
            if c in self.df.columns:
                self.nodes.append(c)
                
        for c in self.macro_cols:
            if c in self.df.columns:
                self.nodes.append(c)
                
        # Remove duplicates while preserving order
        self.nodes = list(dict.fromkeys(self.nodes))
                
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for node, i in self.node_to_idx.items()}
        self.n_nodes = len(self.nodes)
        
    def build_correlation_graph(self, threshold=0.6) -> nx.Graph:
        """Builds a graph where edges represent strong correlations."""
        # Combine price data and macro data
        graph_df = self.df[self.nodes].copy()
        
        # Calculate Pearson correlation matrix
        corr = graph_df.corr().fillna(0).values
        
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
            
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                c = corr[i, j]
                if abs(c) > threshold:
                    G.add_edge(self.nodes[i], self.nodes[j], weight=abs(c), sign=np.sign(c))
                    
        # Add strong domain-knowledge edges regardless of correlation
        macro_sources = ['Petrol (92 Octane)', 'USD_LKR']
        for macro in macro_sources:
            if macro in self.nodes:
                for food in self.food_cols:
                    if food in self.nodes and not G.has_edge(macro, food):
                        G.add_edge(macro, food, weight=0.3, sign=1.0) # Base macroeconomic influence
                        
        if 'Food Index' in self.nodes and 'Inflation' in self.nodes:
            G.add_edge('Food Index', 'Inflation', weight=0.9, sign=1.0)
            
        return G
        
    def get_adjacency_matrix(self, G: nx.Graph) -> np.ndarray:
        """Returns the adjacency matrix for the ST-GNN."""
        A = nx.to_numpy_array(G, nodelist=self.nodes, weight='weight')
        # Add self-loops (identity matrix)
        A = A + np.eye(self.n_nodes)
        # Normalize adjacency (D^-0.5 A D^-0.5)
        rowsum = A.sum(axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_mat_inv_sqrt = np.diag(d_inv_sqrt)
        A_norm = D_mat_inv_sqrt.dot(A).dot(D_mat_inv_sqrt)
        return A_norm
