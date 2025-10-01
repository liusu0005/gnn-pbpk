"""
Dynamic GNN-PBPK Model Implementation

This module implements a Graph Neural Network-based Physiologically Based Pharmacokinetic (PBPK) model
that learns drug concentration dynamics on a physiological graph representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
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
                 edge_features: int = 3,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 temporal_hidden: int = 32,
                 output_dim: int = 1):
        super(PhysiologicalGNN, self).__init__()
        
        self.num_organs = num_organs
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Drug embedding and adaptation
        self.drug_embedding = nn.Embedding(1000, hidden_dim)
        self.drug_adaptation = nn.Linear(hidden_dim, hidden_dim)
        
        # Input projection to handle concatenated features
        # Need to handle both cases: node_features + hidden_dim and hidden_dim + hidden_dim
        self.input_projection_node = nn.Linear(node_features + hidden_dim, hidden_dim)
        self.input_projection_hidden = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        
        # Graph Attention Layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                                      edge_dim=edge_features, concat=True, add_self_loops=False))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                          heads=num_heads, edge_dim=edge_features, concat=True, add_self_loops=False))
        
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                      heads=1, edge_dim=edge_features, concat=False, add_self_loops=False))
        
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
        
        # Message passing functions (learned)
        self.message_function = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
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
            # Handle different drug_features shapes
            if drug_features.dim() == 3:
                drug_features = drug_features.mean(dim=1)
            elif drug_features.dim() == 2 and drug_features.size(0) > 1:
                drug_features = drug_features.mean(dim=0, keepdim=True)
            
            drug_features_expanded = drug_features.expand(x.size(0), -1)
        
        # Concatenate and project features
        x_combined = torch.cat([x, drug_features_expanded], dim=1)
        
        # Choose appropriate projection based on input dimension
        if x.size(1) == self.node_features:
            # Original node features case
            x = self.input_projection_node(x_combined)
        else:
            # Hidden state case (from DynamicGNNPBPK)
            x = self.input_projection_hidden(x_combined)
        
        # Graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
        
        # Custom message passing refinement (residual)
        try:
            refined = self.message_passing_step(x, edge_index, edge_attr)
            x = x + refined
        except Exception:
            pass
        
        # Temporal modeling (if sequence data)
        if x.dim() == 3:
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
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
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
    Dynamic GNN-PBPK model for temporal sequence prediction
    """
    
    def __init__(self, 
                 num_organs: int = 15,
                 node_features: int = 5,
                 edge_features: int = 3,
                 hidden_dim: int = 64,
                 sequence_length: int = 96,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 temporal_hidden: int = 32):
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
            temporal_hidden=temporal_hidden,
            output_dim=hidden_dim  # Output hidden state, not concentration
        )
        
        # Initial concentration embedding
        self.initial_concentration_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Concentration prediction head
        self.concentration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
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
        
        # Convert time_steps to integer if it's a tensor
        if torch.is_tensor(time_steps):
            if time_steps.numel() == 1:
                time_steps = time_steps.item()
            else:
                time_steps = int(time_steps[0])
        else:
            time_steps = int(time_steps)
        
        num_nodes = node_features.size(0)
        
        # Initial hidden state: embed initial concentrations
        current_hidden = self.initial_concentration_embedding(initial_concentrations)
        
        predictions = []
        
        for t in range(time_steps):
            # Apply GNN to current hidden state
            hidden_features = self.base_gnn(
                current_hidden, edge_index, edge_attr, drug_id
            )
            
            # Predict concentration for this time step
            concentration = self.concentration_head(hidden_features)
            predictions.append(concentration)
            
            # Update hidden state for next time step
            current_hidden = hidden_features
        
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
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                batch.node_features,
                batch.edge_index,
                batch.edge_attr,
                batch.drug_id,
                batch.initial_concentrations,
                batch.time_steps
            )
            
            # Compute loss
            # The target should match the predictions shape: [time_steps, num_nodes, 1]
            # batch.concentrations is [batch_size * time_steps, num_nodes] when batched
            batch_size = len(torch.unique(batch.batch)) if hasattr(batch, 'batch') else 1
            time_steps = predictions.size(0)
            num_nodes_per_drug = 15  # Fixed number of organs
            
            # Reshape target to match predictions
            if batch.concentrations.dim() == 2:
                # Reshape from [batch_size * time_steps, num_nodes] to [time_steps, batch_size * num_nodes, 1]
                target = batch.concentrations.view(time_steps, -1, 1)
            else:
                # If already 3D, permute
                target = batch.concentrations.permute(1, 0, 2)
            
            loss = F.mse_loss(predictions, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                predictions = self.model(
                    batch.node_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.drug_id,
                    batch.initial_concentrations,
                    batch.time_steps
                )
                
                # Reshape target to match predictions
                time_steps = predictions.size(0)
                
                if batch.concentrations.dim() == 2:
                    # Reshape from [batch_size * time_steps, num_nodes] to [time_steps, batch_size * num_nodes, 1]
                    target = batch.concentrations.view(time_steps, -1, 1)
                else:
                    # If already 3D, permute
                    target = batch.concentrations.permute(1, 0, 2)
                
                loss = F.mse_loss(predictions, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


def create_physiological_graph_data(concentrations: np.ndarray, drug_properties: np.ndarray, graph_features: np.ndarray) -> List[Data]:
    """Create PyTorch Geometric Data objects for GNN training"""
    num_drugs, time_steps, num_organs = concentrations.shape
    
    # Create a very simple physiological graph structure (same for all drugs)
    # Just connect each organ to the next one in a chain
    edge_list = []
    
    # Create a simple chain graph
    for i in range(num_organs - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])  # Undirected
    
    # Add a few random connections to make it more interesting
    import random
    random.seed(42)
    for _ in range(min(10, num_organs)):
        i = random.randint(0, num_organs - 1)
        j = random.randint(0, num_organs - 1)
        if i != j and [i, j] not in edge_list:
            edge_list.append([i, j])
            edge_list.append([j, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Verify all indices are valid
    print(f"Graph info: {num_organs} nodes, {edge_index.size(1)} edges")
    print(f"Edge index range: {edge_index.min().item()} to {edge_index.max().item()}")
    assert edge_index.max().item() < num_organs, f"Invalid edge index: {edge_index.max().item()} >= {num_organs}"
    
    # Node features (organ properties) - same for all drugs
    node_features = torch.tensor(graph_features, dtype=torch.float32)
    
    # Edge features (flow properties) - match experiment expectation of 2 features
    edge_attr = torch.randn(edge_index.size(1), 2)
    
    # Create Data objects for each drug
    data_list = []
    for drug_idx in range(num_drugs):
        # Drug properties
        drug_props = torch.tensor(drug_properties[drug_idx], dtype=torch.float32)
        
        # Concentration time series
        conc_series = torch.tensor(concentrations[drug_idx], dtype=torch.float32)
        
        # Initial concentrations (first time point)
        initial_conc = conc_series[0:1].unsqueeze(-1)  # [1, num_organs, 1]
        
        # Create Data object with batch information to prevent edge index issues
        data = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_id=torch.tensor([drug_idx], dtype=torch.long),
            drug_properties=drug_props,
            concentrations=conc_series,
            initial_concentrations=initial_conc.squeeze(0),  # [num_organs, 1]
            time_steps=torch.tensor([time_steps], dtype=torch.long),
            # Add batch information to prevent edge index concatenation issues
            batch=torch.zeros(num_organs, dtype=torch.long)  # All nodes belong to batch 0
        )
        data_list.append(data)
    
    return data_list


def create_sample_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample data for testing"""
    num_nodes = 15
    num_edges = 30
    time_steps = 96
    
    # Node features [num_nodes, node_features]
    node_features = torch.randn(num_nodes, 5)
    
    # Edge connectivity [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Edge features [num_edges, edge_features]
    edge_attr = torch.randn(num_edges, 3)
    
    # Drug ID [batch_size]
    drug_id = torch.tensor([0, 1])
    
    # Initial concentrations [num_nodes, 1]
    initial_concentrations = torch.randn(num_nodes, 1)
    
    # Target concentrations [time_steps, num_nodes, 1]
    concentrations = torch.randn(time_steps, num_nodes, 1)
    
    return node_features, edge_index, edge_attr, drug_id, initial_concentrations, concentrations


def main():
    """Test the GNN-PBPK model"""
    print("Testing GNN-PBPK model...")
    
    # Create model
    model = DynamicGNNPBPK(
        num_organs=15,
        node_features=5,
        edge_features=3,
        hidden_dim=64,
        sequence_length=96
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create sample data
    node_features, edge_index, edge_attr, drug_id, initial_concentrations, concentrations = create_sample_data()
    
    # Test forward pass
    try:
        predictions = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            drug_id=drug_id,
            initial_concentrations=initial_concentrations,
            time_steps=96
        )
        
        print(f"Forward pass successful!")
        print(f"Input shapes:")
        print(f"  node_features: {node_features.shape}")
        print(f"  edge_index: {edge_index.shape}")
        print(f"  edge_attr: {edge_attr.shape}")
        print(f"  drug_id: {drug_id.shape}")
        print(f"  initial_concentrations: {initial_concentrations.shape}")
        print(f"Output shape: {predictions.shape}")
        
        # Test trainer
        trainer = GNNPBPKTrainer(model)
        print("Trainer created successfully!")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()