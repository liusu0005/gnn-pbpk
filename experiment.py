"""
Main Experiment Script for GNN-PBPK vs Traditional PBPK Comparison
Generates synthetic data, trains models, and evaluates performance
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import SyntheticDataGenerator, PhysiologicalGraph
from gnn_pbpk_model import DynamicGNNPBPK, GNNPBPKTrainer, create_physiological_graph_data
from traditional_pbpk import TraditionalPBPK, PBPKEnsemble, create_standard_physiological_parameters, DrugParameters

class ExperimentRunner:
    """
    Main experiment runner for comparing GNN-PBPK vs Traditional PBPK
    """
    
    def __init__(self, n_drugs: int = 50, random_seed: int = 42):
        self.n_drugs = n_drugs
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator()
        self.physio_params = create_standard_physiological_parameters()
        self.traditional_pbpk = TraditionalPBPK(self.physio_params)
        self.ensemble_pbpk = PBPKEnsemble(self.physio_params, n_models=5)
        
        # Results storage
        self.results = {}
        
    def generate_dataset(self):
        """Generate synthetic dataset"""
        print("Generating synthetic dataset...")
        
        # Generate data
        concentrations, graph_features, drug_properties = self.data_generator.generate_drug_dataset(self.n_drugs)
        
        # Store data
        self.concentrations = concentrations
        self.graph_features = graph_features
        self.drug_properties = drug_properties
        self.time_points = self.data_generator.time_points
        
        print(f"Dataset generated: {concentrations.shape}")
        return concentrations, graph_features, drug_properties
    
    def prepare_gnn_data(self):
        """Prepare data for GNN training"""
        print("Preparing GNN data...")
        
        # Create PyTorch Geometric data objects
        drug_ids = np.arange(self.n_drugs)
        gnn_data = create_physiological_graph_data(
            self.concentrations, self.graph_features, drug_ids
        )
        
        # Split data
        train_data, test_data = train_test_split(gnn_data, test_size=0.3, random_state=self.random_seed)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=self.random_seed)
        
        # Create data loaders
        self.train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def train_gnn_model(self, epochs: int = 100):
        """Train the GNN-PBPK model"""
        print("Training GNN-PBPK model...")
        
        # Create model
        self.gnn_model = DynamicGNNPBPK(
            num_organs=15,
            node_features=5,
            edge_features=2,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            temporal_hidden=128,
            sequence_length=len(self.time_points)
        )
        
        # Create trainer
        self.gnn_trainer = GNNPBPKTrainer(self.gnn_model, learning_rate=0.001)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.gnn_trainer.train_epoch(self.train_loader)
            val_loss = self.gnn_trainer.validate(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        self.gnn_train_losses = train_losses
        self.gnn_val_losses = val_losses
        
        print("GNN training completed!")
        return train_losses, val_losses
    
    def evaluate_traditional_pbpk(self):
        """Evaluate traditional PBPK model"""
        print("Evaluating Traditional PBPK model...")
        
        traditional_predictions = []
        ensemble_predictions = []
        
        for i, drug_props in enumerate(self.drug_properties):
            # Convert to DrugParameters format
            drug_params = DrugParameters(
                clearance=drug_props.clearance,
                volume_distribution=drug_props.volume_distribution,
                fu_plasma=drug_props.fu_plasma,
                molecular_weight=drug_props.molecular_weight,
                log_p=drug_props.log_p,
                tissue_affinity=drug_props.tissue_affinity,
                metabolic_rate=drug_props.metabolic_rate,
                transporter_mediated=drug_props.transporter_mediated
            )
            
            # Traditional PBPK prediction
            trad_pred = self.traditional_pbpk.simulate_drug_kinetics(
                drug_params, time_points=self.time_points
            )
            traditional_predictions.append(trad_pred)
            
            # Ensemble PBPK prediction
            ensemble_mean, _ = self.ensemble_pbpk.predict(
                drug_params, time_points=self.time_points
            )
            ensemble_predictions.append(ensemble_mean)
        
        self.traditional_predictions = np.array(traditional_predictions)
        self.ensemble_predictions = np.array(ensemble_predictions)
        
        print("Traditional PBPK evaluation completed!")
        return self.traditional_predictions, self.ensemble_predictions
    
    def evaluate_gnn_model(self):
        """Evaluate the trained GNN model"""
        print("Evaluating GNN-PBPK model...")
        
        self.gnn_model.eval()
        gnn_predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Get predictions
                pred = self.gnn_model(
                    node_features=batch.node_features,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    drug_id=batch.drug_id,
                    initial_concentrations=batch.initial_concentrations,
                    time_steps=batch.sequence_length
                )
                
                # Convert to numpy and store
                pred_np = pred.squeeze().cpu().numpy()
                gnn_predictions.append(pred_np)
        
        self.gnn_predictions = np.concatenate(gnn_predictions, axis=1)
        print("GNN evaluation completed!")
        return self.gnn_predictions
    
    def compute_metrics(self):
        """Compute performance metrics for all models"""
        print("Computing performance metrics...")
        
        # Get test data
        test_indices = range(int(0.7 * self.n_drugs), self.n_drugs)
        test_concentrations = self.concentrations[test_indices]
        test_traditional = self.traditional_predictions[test_indices]
        test_ensemble = self.ensemble_predictions[test_indices]
        
        # Ensure GNN predictions match test data shape
        if hasattr(self, 'gnn_predictions'):
            test_gnn = self.gnn_predictions
        else:
            test_gnn = test_traditional  # Placeholder if GNN not evaluated
        
        # Compute metrics for each model
        models = {
            'Traditional PBPK': test_traditional,
            'Ensemble PBPK': test_ensemble,
            'GNN-PBPK': test_gnn
        }
        
        metrics = {}
        
        for model_name, predictions in models.items():
            # Flatten for overall metrics
            y_true = test_concentrations.flatten()
            y_pred = predictions.flatten()
            
            # Remove zeros to avoid division by zero in MAPE
            mask = y_true > 0
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]
            
            model_metrics = {
                'MAPE': mean_absolute_percentage_error(y_true_masked, y_pred_masked) * 100,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MAE': np.mean(np.abs(y_true - y_pred))
            }
            
            # Organ-specific metrics
            organ_metrics = {}
            organs = ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 
                     'fat', 'lung', 'spleen', 'gut', 'bone', 'skin', 'pancreas',
                     'adrenal', 'thyroid']
            
            for i, organ in enumerate(organs):
                if i < test_concentrations.shape[2]:
                    org_true = test_concentrations[:, :, i].flatten()
                    org_pred = predictions[:, :, i].flatten()
                    
                    mask = org_true > 0
                    if mask.sum() > 0:
                        organ_metrics[organ] = {
                            'MAPE': mean_absolute_percentage_error(org_true[mask], org_pred[mask]) * 100,
                            'R2': r2_score(org_true, org_pred)
                        }
            
            metrics[model_name] = {
                'overall': model_metrics,
                'organs': organ_metrics
            }
        
        self.metrics = metrics
        return metrics
    
    def generate_results_summary(self):
        """Generate comprehensive results summary"""
        print("Generating results summary...")
        
        # Overall performance comparison
        overall_results = []
        for model_name, model_metrics in self.metrics.items():
            overall_results.append({
                'Model': model_name,
                'MAPE (%)': model_metrics['overall']['MAPE'],
                'RMSE': model_metrics['overall']['RMSE'],
                'R²': model_metrics['overall']['R2'],
                'MAE': model_metrics['overall']['MAE']
            })
        
        overall_df = pd.DataFrame(overall_results)
        
        # Organ-specific performance
        organ_results = []
        for model_name, model_metrics in self.metrics.items():
            for organ, organ_metrics in model_metrics['organs'].items():
                organ_results.append({
                    'Model': model_name,
                    'Organ': organ,
                    'MAPE (%)': organ_metrics['MAPE'],
                    'R²': organ_metrics['R2']
                })
        
        organ_df = pd.DataFrame(organ_results)
        
        # Performance improvement analysis
        gnn_mape = self.metrics['GNN-PBPK']['overall']['MAPE']
        trad_mape = self.metrics['Traditional PBPK']['overall']['MAPE']
        improvement = ((trad_mape - gnn_mape) / trad_mape) * 100
        
        summary = {
            'overall_performance': overall_df,
            'organ_performance': organ_df,
            'improvement_percentage': improvement,
            'gnn_vs_traditional': {
                'GNN MAPE': gnn_mape,
                'Traditional MAPE': trad_mape,
                'Improvement': improvement
            }
        }
        
        self.results_summary = summary
        return summary
    
    def run_full_experiment(self):
        """Run the complete experiment"""
        print("=" * 60)
        print("DYNAMIC GNN-PBPK vs TRADITIONAL PBPK EXPERIMENT")
        print("=" * 60)
        
        # Step 1: Generate dataset
        self.generate_dataset()
        
        # Step 2: Prepare GNN data
        self.prepare_gnn_data()
        
        # Step 3: Train GNN model
        self.train_gnn_model(epochs=50)  # Reduced for demo
        
        # Step 4: Evaluate traditional PBPK
        self.evaluate_traditional_pbpk()
        
        # Step 5: Evaluate GNN model
        self.evaluate_gnn_model()
        
        # Step 6: Compute metrics
        self.compute_metrics()
        
        # Step 7: Generate summary
        self.generate_results_summary()
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self.results_summary

def main():
    """Run the main experiment"""
    
    # Create and run experiment
    experiment = ExperimentRunner(n_drugs=30, random_seed=42)  # Reduced for demo
    results = experiment.run_full_experiment()
    
    # Print results
    print("\nOVERALL PERFORMANCE COMPARISON:")
    print(results['overall_performance'].to_string(index=False))
    
    print(f"\nPERFORMANCE IMPROVEMENT:")
    print(f"GNN-PBPK vs Traditional PBPK: {results['improvement_percentage']:.1f}% reduction in MAPE")
    
    # Save results
    results['overall_performance'].to_csv('overall_performance.csv', index=False)
    results['organ_performance'].to_csv('organ_performance.csv', index=False)
    
    print("\nResults saved to CSV files!")
    
    return results

if __name__ == "__main__":
    results = main()
