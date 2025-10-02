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
    Simple GNN component for DynamicGNNPBPK - just basic GAT layers
    """
    
    def __init__(self, 
                 num_organs: int = 15,
                 node_features: int = 5,
                 edge_features: int = 3,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 3):
        super(PhysiologicalGNN, self).__init__()
        
        self.num_organs = num_organs
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Simple GAT layers only
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                                              edge_dim=edge_features, concat=True, add_self_loops=False))
            else:
                self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                              heads=num_heads, edge_dim=edge_features, concat=True, add_self_loops=False))
        
        # Final GAT layer (single head)
        self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                      heads=1, edge_dim=edge_features, concat=False, add_self_loops=False))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Simple forward pass through GAT layers only"""
        
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index, edge_attr)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
        
        return x


class DynamicGNNPBPK(nn.Module):
    """
    Dynamic GNN-PBPK model implementing the complete 6-layer architecture:
    1. Input
    2. Graph Construction (node and edge features)
    3. GAT Layers (including Drug Embedding)
    4. Message Passing
    5. Temporal LSTM
    6. Output
    """
    
    def __init__(self, 
                 num_organs: int = 15,
                 drug_features: int = 8,  # clearance, volume_distribution, fu_plasma, molecular_weight, log_p, tissue_affinity, metabolic_rate, transporter_mediated
                 physio_features: int = 31,  # 15 organ_volumes + 15 blood_flows + 1 cardiac_output
                 edge_features: int = 3,
                 hidden_dim: int = 64,
                 sequence_length: int = 96,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 temporal_hidden: int = 32):
        super(DynamicGNNPBPK, self).__init__()
        
        self.num_organs = num_organs
        self.drug_features = drug_features
        self.physio_features = physio_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Layer 1: Input Processing - Same inputs as traditional PBPK
        # Drug parameters processing
        self.drug_input_processor = nn.Sequential(
            nn.Linear(drug_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Physiological parameters processing
        self.physio_input_processor = nn.Sequential(
            nn.Linear(physio_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine drug and physiological features
        self.input_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Layer 2: Graph Construction - Use node and edge features to create graph, then flow Layer 1 input through it
        # Node features: organ-specific properties (volumes, flows, etc.) - define graph structure
        self.node_feature_processor = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5 node features per organ
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge features: flow properties between organs - define graph connections
        self.edge_feature_processor = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, edge_features)
        )
        
        # Graph flow layer: Layer 1 input flows through the graph structure
        self.graph_flow = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer 3: GAT Layers (drug features already processed in Layer 1)
        self.gat_component = PhysiologicalGNN(
            num_organs=num_organs,
            node_features=hidden_dim,  # After projection
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Layer 4: Message Passing
        self.message_function = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_update_function = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer 5: Temporal LSTM
        self.temporal_lstm = nn.LSTM(hidden_dim, temporal_hidden, batch_first=True)
        self.temporal_linear = nn.Linear(temporal_hidden, hidden_dim)
        
        # Layer 6: Output
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Scale to [0, 1] range
        )
        
    def forward(self, 
                drug_params: torch.Tensor,
                physio_params: torch.Tensor,
                dose: torch.Tensor,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                time_points: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the 6-layer architecture using same inputs as traditional PBPK:
        1. Input → 2. Graph Construction → 3. GAT Layers → 4. Message Passing → 5. Temporal LSTM → 6. Output
        
        Args:
            drug_params: Drug parameters [batch_size, drug_features] - clearance, volume_distribution, fu_plasma, molecular_weight, log_p, tissue_affinity, metabolic_rate, transporter_mediated
            physio_params: Physiological parameters [batch_size, physio_features] - organ_volumes, blood_flows, cardiac_output
            dose: Drug dose [batch_size] - mg/kg
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            time_points: Time points for simulation [time_steps]
            
        Returns:
            Predicted concentration sequences [time_steps, num_organs, 1]
        """
        
        if time_points is None:
            time_steps = self.sequence_length
        else:
            time_steps = time_points.size(0)
        
        # Convert time_steps to integer if it's a tensor
        if torch.is_tensor(time_steps):
            if time_steps.numel() == 1:
                time_steps = time_steps.item()
            else:
                time_steps = int(time_steps[0])
        else:
            time_steps = int(time_steps)
        
        # Layer 1: Input Processing - Same inputs as traditional PBPK
        # Process drug parameters
        drug_features = self.drug_input_processor(drug_params)  # [batch_size, hidden_dim]
        
        # Process physiological parameters
        physio_features = self.physio_input_processor(physio_params)  # [batch_size, hidden_dim]
        
        # Combine drug and physiological features
        combined_features = torch.cat([drug_features, physio_features], dim=1)  # [batch_size, hidden_dim * 2]
        input_features = self.input_combiner(combined_features)  # [batch_size, hidden_dim]
        
        # Layer 2: Graph Construction - Use node and edge features to create graph, then flow Layer 1 input through it
        batch_size = drug_params.size(0)
        
        # Process node features (organ-specific properties) - define graph structure
        if node_features.dim() == 1:
            # Handle flattened node features: [batch_size * num_organs * 5] -> [batch_size * num_organs, 5]
            node_features = node_features.view(batch_size * self.num_organs, 5)
        elif node_features.dim() == 2 and node_features.size(0) > self.num_organs:
            # Handle batched data: [batch_size * num_organs, 5]
            pass  # Already in correct shape
        
        # Process node features to define graph structure
        node_processed = self.node_feature_processor(node_features)  # [batch_size * num_organs, hidden_dim]
        
        # Process edge features to define graph connections
        edge_processed = self.edge_feature_processor(edge_attr)  # [num_edges, edge_features]
        
        # Expand Layer 1 input to all nodes and flow through graph structure
        input_expanded = input_features.unsqueeze(1).expand(-1, self.num_organs, -1)  # [batch_size, num_organs, hidden_dim]
        input_expanded = input_expanded.reshape(-1, self.hidden_dim)  # [batch_size * num_organs, hidden_dim]
        
        # Layer 1 input flows through the graph structure defined by node and edge features
        x = self.graph_flow(input_expanded)  # [batch_size * num_organs, hidden_dim]
        
        # Combine with node features (graph structure influences the flow)
        x = x + node_processed  # [batch_size * num_organs, hidden_dim]
        
        # Layer 3: GAT Layers (drug features already integrated in Layer 1)
        x = self.gat_component(x, edge_index, edge_processed)
        
        # Layer 4: Message Passing
        x = self.message_passing_step(x, edge_index, edge_processed)
        
        # Layer 5: Temporal LSTM
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x, _ = self.temporal_lstm(x)
        x = self.temporal_linear(x)
        x = x.squeeze(0)
        
        # Layer 6: Output
        concentrations = self.output_layer(x)
        
        # Reshape back to [batch_size, num_organs, 1]
        batch_size = drug_params.size(0)
        concentrations = concentrations.view(batch_size, self.num_organs, 1)
        
        # For temporal prediction, create sequence
        predictions = []
        current_state = x.view(batch_size, self.num_organs, -1)
        
        for t in range(time_steps):
            # Predict concentration for this time step
            concentration = self.output_layer(current_state.view(-1, self.hidden_dim))
            concentration = concentration.view(batch_size, self.num_organs, 1)
            predictions.append(concentration)
            
            # Update state for next time step (simple decay)
            current_state = current_state * 0.9 + x.view(batch_size, self.num_organs, -1) * 0.1
        
        predictions = torch.stack(predictions, dim=0)  # [time_steps, batch_size, num_organs, 1]
        return predictions
    
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
            
            # Convert old batch format to new format
            # Extract drug parameters from batch
            batch_size = len(torch.unique(batch.batch)) if hasattr(batch, 'batch') else 1
            
            # Create dummy drug and physiological parameters for now
            drug_params = torch.randn(batch_size, 8)  # 8 drug features
            physio_params = torch.randn(batch_size, 31)  # 31 physiological features
            dose = torch.ones(batch_size) * 0.1  # Default dose
            time_points = torch.linspace(0, 24, 96)  # Default time points
            
            # Forward pass with new architecture
            predictions = self.model(
                drug_params=drug_params,
                physio_params=physio_params,
                dose=dose,
                node_features=batch.node_features,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                time_points=time_points
            )
            
            # Compute loss
            # Predictions shape: [time_steps, batch_size, num_organs, 1]
            # Target shape: [time_steps, batch_size, num_organs, 1]
            time_steps = predictions.size(0)
            batch_size = predictions.size(1)
            num_organs = predictions.size(2)
            
            # Reshape target to match predictions
            if batch.concentrations.dim() == 2:
                # Reshape from [batch_size * time_steps, num_organs] to [time_steps, batch_size, num_organs, 1]
                target = batch.concentrations.view(time_steps, batch_size, num_organs, 1)
            else:
                # If already 3D, add channel dimension
                target = batch.concentrations.unsqueeze(-1)
            
            # Use a combination of MSE and MAE for better training
            mse_loss = F.mse_loss(predictions, target)
            mae_loss = F.l1_loss(predictions, target)
            loss = mse_loss + 0.1 * mae_loss
            
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
                # Convert old batch format to new format
                batch_size = len(torch.unique(batch.batch)) if hasattr(batch, 'batch') else 1
                
                # Create dummy drug and physiological parameters for now
                drug_params = torch.randn(batch_size, 8)  # 8 drug features
                physio_params = torch.randn(batch_size, 31)  # 31 physiological features
                dose = torch.ones(batch_size) * 0.1  # Default dose
                time_points = torch.linspace(0, 24, 96)  # Default time points
                
                predictions = self.model(
                    drug_params=drug_params,
                    physio_params=physio_params,
                    dose=dose,
                    node_features=batch.node_features,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    time_points=time_points
                )
                
                # Reshape target to match predictions
                time_steps = predictions.size(0)
                batch_size = predictions.size(1)
                num_organs = predictions.size(2)
                
                if batch.concentrations.dim() == 2:
                    # Reshape from [batch_size * time_steps, num_organs] to [time_steps, batch_size, num_organs, 1]
                    target = batch.concentrations.view(time_steps, batch_size, num_organs, 1)
                else:
                    # If already 3D, add channel dimension
                    target = batch.concentrations.unsqueeze(-1)
                
                # Use a combination of MSE and MAE for better training
                mse_loss = F.mse_loss(predictions, target)
                mae_loss = F.l1_loss(predictions, target)
                loss = mse_loss + 0.1 * mae_loss
                
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
    # graph_features is now (n_organs, 5) instead of flattened
    base_node_features = torch.tensor(graph_features[0], dtype=torch.float32)  # Use first drug's features as template
    
    # Edge features (flow properties) - match GNN expectation of 3 features
    edge_attr = torch.randn(edge_index.size(1), 3)
    
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
            node_features=base_node_features,  # [num_organs, 5]
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


def create_sample_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample data for testing with same inputs as traditional PBPK"""
    batch_size = 2
    num_organs = 15
    num_edges = 30
    time_steps = 96
    
    # Drug parameters [batch_size, drug_features] - clearance, volume_distribution, fu_plasma, molecular_weight, log_p, tissue_affinity, metabolic_rate, transporter_mediated
    drug_params = torch.randn(batch_size, 8)
    
    # Physiological parameters [batch_size, physio_features] - 15 organ_volumes + 15 blood_flows + 1 cardiac_output
    physio_params = torch.randn(batch_size, 31)
    
    # Dose [batch_size] - mg/kg
    dose = torch.tensor([0.1, 0.2])
    
    # Node features [batch_size * num_organs, 5] - organ-specific properties
    node_features = torch.randn(batch_size * num_organs, 5)
    
    # Edge connectivity [2, num_edges]
    edge_index = torch.randint(0, num_organs, (2, num_edges))
    
    # Edge features [num_edges, edge_features]
    edge_attr = torch.randn(num_edges, 3)
    
    # Time points [time_steps]
    time_points = torch.linspace(0, 24, time_steps)
    
    return drug_params, physio_params, dose, node_features, edge_index, edge_attr, time_points


def main():
    """Test the GNN-PBPK model"""
    print("Testing GNN-PBPK model...")
    
    # Create model
    model = DynamicGNNPBPK(
        num_organs=15,
        drug_features=8,
        physio_features=31,
        edge_features=3,
        hidden_dim=64,
        sequence_length=96
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create sample data
    drug_params, physio_params, dose, node_features, edge_index, edge_attr, time_points = create_sample_data()
    
    # Test forward pass
    try:
        predictions = model(
            drug_params=drug_params,
            physio_params=physio_params,
            dose=dose,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            time_points=time_points
        )
        
        print(f"Forward pass successful!")
        print(f"Input shapes:")
        print(f"  drug_params: {drug_params.shape}")
        print(f"  physio_params: {physio_params.shape}")
        print(f"  dose: {dose.shape}")
        print(f"  node_features: {node_features.shape}")
        print(f"  edge_index: {edge_index.shape}")
        print(f"  edge_attr: {edge_attr.shape}")
        print(f"  time_points: {time_points.shape}")
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