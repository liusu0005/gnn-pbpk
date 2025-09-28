"""
Dynamic GNN-PBPK Model Implementation
Graph Neural Network for Physiological-Based Pharmacokinetic Modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional

class PhysiologicalGNN(nn.Module):
    """
    Graph Neural Network for PBPK modeling with temporal dynamics
    """
    
    def __init__(self, 
                 num_organs: int = 15,
                 node_features: int = 5,
                 edge_features: int = 2,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 temporal_hidden: int = 128,
                 output_dim: int = 1):
        super(PhysiologicalGNN, self).__init__()
        
        self.num_organs = num_organs
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Graph Attention Layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(node_features, hidden_dim, heads=num_heads, 
                                      edge_dim=edge_features, concat=True))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                          heads=num_heads, edge_dim=edge_features, concat=True))
        
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                      heads=1, edge_dim=edge_features, concat=False))
        
        # Temporal modeling layers
        self.temporal_lstm = nn.LSTM(hidden_dim, temporal_hidden, batch_first=True)
        self.temporal_linear = nn.Linear(temporal_hidden, hidden_dim)
        
        # Output layers for concentration prediction
        self.concentration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Drug-specific adaptation layers
        self.drug_embedding = nn.Embedding(1000, hidden_dim)  # Support up to 1000 drugs
        self.drug_adaptation = nn.Linear(hidden_dim, hidden_dim)
        
        # Message passing functions (learned)
        self.message_function = nn.Sequential(
            nn.Linear(hidden_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update function (learned)
        self.node_update_function = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, drug_id: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN-PBPK model
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            drug_id: Drug identifier [batch_size]
            batch: Batch assignment for nodes
            
        Returns:
            Predicted concentrations [num_nodes, output_dim]
        """
        
        # Drug-specific adaptation
        drug_embedding = self.drug_embedding(drug_id)
        drug_features = self.drug_adaptation(drug_embedding)
        
        # Apply drug features to all nodes
        if batch is not None:
            drug_features_expanded = drug_features[batch]
        else:
            drug_features_expanded = drug_features.unsqueeze(0).expand(x.size(0), -1)
        
        x = x + drug_features_expanded
        
        # Graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
        
        # Temporal modeling (if sequence data)
        if x.dim() == 3:  # [batch, time, features]
            x, _ = self.temporal_lstm(x)
            x = self.temporal_linear(x)
        
        # Predict concentrations
        concentrations = self.concentration_predictor(x)
        
        return concentrations
    
    def message_passing_step(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Custom message passing step for drug concentration flow
        """
        row, col = edge_index
        
        # Create messages
        edge_features = torch.cat([x[row], edge_attr], dim=1)
        messages = self.message_function(edge_features)
        
        # Aggregate messages
        aggregated = torch.zeros_like(x)
        for i in range(x.size(0)):
            mask = (col == i)
            if mask.sum() > 0:
                aggregated[i] = messages[mask].mean(dim=0)
        
        # Update nodes
        updated_features = torch.cat([x, aggregated], dim=1)
        x_new = self.node_update_function(updated_features)
        
        return x_new

class DynamicGNNPBPK(nn.Module):
    """
    Dynamic GNN-PBPK model that learns temporal evolution of drug concentrations
    """
    
    def __init__(self, 
                 num_organs: int = 15,
                 node_features: int = 5,
                 edge_features: int = 2,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 temporal_hidden: int = 128,
                 sequence_length: int = 96):  # 48 hours * 2 (0.5h intervals)
        super(DynamicGNNPBPK, self).__init__()
        
        self.num_organs = num_organs
        self.sequence_length = sequence_length
        
        # Base GNN for single time step
        self.base_gnn = PhysiologicalGNN(
            num_organs=num_organs,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            temporal_hidden=temporal_hidden
        )
        
        # Temporal evolution network
        self.temporal_evolution = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Concentration prediction head
        self.concentration_head = nn.Sequential(
            nn.Linear(temporal_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initial concentration embedding
        self.initial_concentration_embedding = nn.Linear(1, hidden_dim)
        
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                drug_id: torch.Tensor,
                initial_concentrations: torch.Tensor,
                time_steps: int = None) -> torch.Tensor:
        """
        Forward pass for temporal sequence prediction
        
        Args:
            node_features: Static node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            drug_id: Drug identifier [batch_size]
            initial_concentrations: Initial drug concentrations [num_nodes, 1]
            time_steps: Number of time steps to predict
            
        Returns:
            Predicted concentration sequences [time_steps, num_nodes, 1]
        """
        
        if time_steps is None:
            time_steps = self.sequence_length
        
        batch_size = drug_id.size(0)
        num_nodes = node_features.size(0)
        
        # Initialize hidden state with initial concentrations
        hidden_state = self.initial_concentration_embedding(initial_concentrations)
        
        # Store predictions
        predictions = []
        current_hidden = hidden_state
        
        for t in range(time_steps):
            # Apply GNN to current state
            gnn_output = self.base_gnn(
                current_hidden, edge_index, edge_attr, drug_id
            )
            
            # Update hidden state
            current_hidden = gnn_output
            
            # Predict concentration for this time step
            concentration = self.concentration_head(current_hidden)
            predictions.append(concentration)
        
        # Stack predictions: [time_steps, num_nodes, 1]
        predictions = torch.stack(predictions, dim=0)
        
        return predictions

class GNNPBPKTrainer:
    """
    Trainer class for the GNN-PBPK model
    """
    
    def __init__(self, model: DynamicGNNPBPK, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader, device: str = 'cpu') -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                node_features=batch.node_features,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                drug_id=batch.drug_id,
                initial_concentrations=batch.initial_concentrations,
                time_steps=batch.sequence_length
            )
            
            # Compute loss
            loss = self.criterion(predictions, batch.target_concentrations)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, device: str = 'cpu') -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                
                predictions = self.model(
                    node_features=batch.node_features,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    drug_id=batch.drug_id,
                    initial_concentrations=batch.initial_concentrations,
                    time_steps=batch.sequence_length
                )
                
                loss = self.criterion(predictions, batch.target_concentrations)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

def create_physiological_graph_data(concentrations: np.ndarray, 
                                  graph_features: np.ndarray,
                                  drug_ids: np.ndarray) -> List[Data]:
    """
    Convert synthetic data to PyTorch Geometric Data objects
    
    Args:
        concentrations: Drug concentrations [n_drugs, time_steps, n_organs]
        graph_features: Graph features [n_drugs, feature_dim]
        drug_ids: Drug identifiers [n_drugs]
        
    Returns:
        List of Data objects for PyTorch Geometric
    """
    
    n_drugs, time_steps, n_organs = concentrations.shape
    
    # Create adjacency matrix (same for all drugs)
    organs = ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 
              'fat', 'lung', 'spleen', 'gut', 'bone', 'skin', 'pancreas',
              'adrenal', 'thyroid']
    
    # Create edge index (simplified - all organs connected to plasma)
    edge_index = []
    for i in range(1, n_organs):  # Skip plasma (index 0)
        edge_index.append([0, i])  # plasma -> organ
        edge_index.append([i, 0])  # organ -> plasma
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # Create edge attributes
    edge_attr = torch.ones(edge_index.size(1), 2)  # Simplified edge features
    
    data_list = []
    
    for drug_idx in range(n_drugs):
        # Node features (extract from graph_features)
        node_features = torch.tensor(graph_features[drug_idx][:n_organs*5]).reshape(n_organs, 5)
        
        # Target concentrations
        target_concentrations = torch.tensor(concentrations[drug_idx]).float()
        
        # Initial concentrations
        initial_concentrations = target_concentrations[0:1].unsqueeze(-1)  # First time step
        
        # Create Data object
        data = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_id=torch.tensor([drug_ids[drug_idx]]),
            target_concentrations=target_concentrations,
            initial_concentrations=initial_concentrations,
            sequence_length=time_steps
        )
        
        data_list.append(data)
    
    return data_list

def main():
    """Test the GNN-PBPK model"""
    print("Testing GNN-PBPK model...")
    
    # Create model
    model = DynamicGNNPBPK(
        num_organs=15,
        node_features=5,
        edge_features=2,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        temporal_hidden=128,
        sequence_length=96
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    num_organs = 15
    time_steps = 96
    
    node_features = torch.randn(num_organs, 5)
    edge_index = torch.randint(0, num_organs, (2, 20))
    edge_attr = torch.randn(20, 2)
    drug_id = torch.randint(0, 50, (batch_size,))
    initial_concentrations = torch.randn(num_organs, 1)
    
    with torch.no_grad():
        predictions = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_id=drug_id,
            initial_concentrations=initial_concentrations,
            time_steps=time_steps
        )
    
    print(f"Output shape: {predictions.shape}")
    print("GNN-PBPK model test completed successfully!")

if __name__ == "__main__":
    main()
