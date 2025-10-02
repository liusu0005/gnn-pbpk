#!/usr/bin/env python3
"""
Experiment runner using the fixed data generator
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from scipy import stats
import json
import os
from typing import Dict, List, Tuple, Any

# Import our modules
from data_generator import SyntheticDataGenerator, DrugProperties
from gnn_pbpk_model import DynamicGNNPBPK, PhysiologicalGNN, GNNPBPKTrainer, create_physiological_graph_data
from traditional_pbpk import TraditionalPBPK, PBPKEnsemble, PhysiologicalParameters, DrugParameters

class ExperimentRunner:
    """Experiment runner for GNN-PBPK model"""
    
    def __init__(self, n_drugs: int = 30, n_epochs: int = 100, force_retrain: bool = False):
        self.n_drugs = n_drugs
        self.n_epochs = n_epochs
        self.force_retrain = force_retrain
        
        # Initialize data generator
        self.data_generator = SyntheticDataGenerator()
        
        # Data storage
        self.concentrations = None
        self.graph_features = None
        self.drug_properties = None
        
        # Model storage
        self.gnn_model = None
        self.gnn_trainer = None
        self.traditional_model = None
        self.ensemble_model = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Results
        self.results = {}
        
    def generate_data(self):
        """Generate synthetic dataset"""
        print("Generating synthetic dataset...")
        
        self.concentrations, self.graph_features, self.drug_properties = \
            self.data_generator.generate_drug_dataset(n_drugs=self.n_drugs)
        
        print(f"Dataset generated: {self.concentrations.shape}")
        print(f"Concentration range: [{self.concentrations.min():.3f}, {self.concentrations.max():.3f}]")
        print(f"Concentration mean: {self.concentrations.mean():.3f}")
        
        return self.concentrations, self.graph_features, self.drug_properties
    
    def _convert_drug_properties(self, drug_props: DrugProperties) -> DrugParameters:
        """Convert DrugProperties to DrugParameters for traditional PBPK models"""
        return DrugParameters(
            clearance=drug_props.clearance,
            volume_distribution=1.0,  # Placeholder
            fu_plasma=drug_props.fu_plasma,
            molecular_weight=drug_props.molecular_weight,
            log_p=drug_props.log_p,
            tissue_affinity=drug_props.tissue_affinity,
            metabolic_rate=drug_props.metabolic_rate,
            transporter_mediated=False
        )
    
    def prepare_gnn_data(self):
        """Prepare data for GNN training"""
        print("Preparing GNN data...")
        
        # Create graph data for each drug
        graph_data_list = []
        
        for i in range(self.n_drugs):
            # Get drug-specific data
            drug_concentrations = self.concentrations[i]  # Shape: (time_steps, num_organs)
            drug_props = self.drug_properties[i]
            drug_graph_features = self.graph_features[i]
            
            # Convert drug properties to tensor
            drug_props_tensor = torch.tensor([
                drug_props.molecular_weight,
                drug_props.log_p,
                drug_props.fu_plasma,
                drug_props.clearance,
                1.0  # volume_distribution (placeholder)
            ], dtype=torch.float32)
            
            # Create graph data - need to reshape for the function
            # The function expects (num_drugs, time_steps, num_organs), so we add a batch dimension
            drug_concentrations_3d = drug_concentrations.reshape(1, drug_concentrations.shape[0], drug_concentrations.shape[1])
            
            graph_data = create_physiological_graph_data(
                concentrations=drug_concentrations_3d,
                drug_properties=[drug_props_tensor],  # List with single drug tensor
                graph_features=drug_graph_features.reshape(1, -1)  # Add batch dimension
            )
            graph_data_list.append(graph_data[0])  # Extract the single graph data object
        
        print(f"Created {len(graph_data_list)} graph data objects")
        
        # Split data
        n_train = int(0.7 * self.n_drugs)
        n_val = int(0.15 * self.n_drugs)
        n_test = self.n_drugs - n_train - n_val
        
        train_data = graph_data_list[:n_train]
        val_data = graph_data_list[n_train:n_train + n_val]
        test_data = graph_data_list[n_train + n_val:]
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Create data loaders
        self.train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        return train_data, val_data, test_data
    
    def initialize_models(self):
        """Initialize all models"""
        print("Initializing models...")
        
        # GNN model parameters
        num_organs = len(self.data_generator.graph.organs)
        time_steps = self.concentrations.shape[1]
        drug_features_dim = 5  # molecular_weight, log_p, fu_plasma, clearance, volume_distribution
        hidden_dim = 64
        temporal_hidden = 32
        
        # Initialize GNN model
        self.gnn_model = DynamicGNNPBPK(
            num_organs=num_organs,
            node_features=drug_features_dim,
            edge_features=3,  # From the graph creation
            hidden_dim=hidden_dim,
            sequence_length=time_steps,
            temporal_hidden=temporal_hidden
        )
        
        # Initialize GNN trainer
        self.gnn_trainer = GNNPBPKTrainer(
            model=self.gnn_model,
            learning_rate=0.001
        )
        
        # Initialize traditional PBPK model
        # Create physiological parameters
        phys_params = PhysiologicalParameters(
            organ_volumes={organ: self.data_generator.graph.volumes[organ] 
                          for organ in self.data_generator.graph.organs},
            blood_flows={organ: self.data_generator.graph.blood_flows[organ] 
                        for organ in self.data_generator.graph.organs},
            cardiac_output=5.0
        )
        
        self.traditional_model = TraditionalPBPK(phys_params)
        self.ensemble_model = PBPKEnsemble(phys_params, n_models=5)
        
        print(f"GNN model parameters: {sum(p.numel() for p in self.gnn_model.parameters()):,}")
        
    def train_gnn_model(self):
        """Train the GNN model or load existing trained model"""
        model_path = 'trained_gnn_model.pt'
        
        # Check if trained model already exists and force_retrain is False
        if os.path.exists(model_path) and not self.force_retrain:
            print(f"Loading existing trained model from {model_path}...")
            try:
                self.gnn_model.load_state_dict(torch.load(model_path))
                print("✅ Successfully loaded trained model!")
                return
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                print("Training new model...")
        elif self.force_retrain:
            print("🔄 Force retrain enabled - training new model...")
        
        print(f"Training GNN model for {self.n_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            train_loss = self.gnn_trainer.train_epoch(self.train_loader)
            
            # Validation
            val_loss = self.gnn_trainer.validate(self.val_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.gnn_model.state_dict(), 'best_gnn_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and save final trained model
        self.gnn_model.load_state_dict(torch.load('best_gnn_model.pt'))
        torch.save(self.gnn_model.state_dict(), model_path)
        print(f"✅ Training completed! Model saved to {model_path}")
        
    def evaluate_models(self):
        """Evaluate all models on test data"""
        print("Evaluating models...")
        
        # Get test data
        test_concentrations = self.concentrations[-len(self.test_loader.dataset):]
        
        # Evaluate GNN model
        gnn_predictions = []
        self.gnn_model.eval()
        
        with torch.no_grad():
            for batch in self.test_loader:
                predictions = self.gnn_model(
                    batch.node_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.drug_id,
                    batch.initial_concentrations,
                    batch.time_steps
                )
                # Reshape predictions to [batch_size, time_steps, num_organs]
                batch_size = len(torch.unique(batch.batch)) if hasattr(batch, 'batch') else 1
                time_steps = predictions.size(0)
                num_organs = 15
                
                # Reshape from [time_steps, batch_size * num_organs, 1] to [batch_size, time_steps, num_organs]
                pred_reshaped = predictions.squeeze(-1).view(time_steps, batch_size, num_organs).permute(1, 0, 2)
                gnn_predictions.append(pred_reshaped.cpu().numpy())
        
        gnn_predictions = np.concatenate(gnn_predictions, axis=0)
        
        # Evaluate traditional PBPK model
        traditional_predictions = []
        for i in range(len(test_concentrations)):
            drug_props = self.drug_properties[-len(test_concentrations) + i]
            drug_params = self._convert_drug_properties(drug_props)
            pred = self.traditional_model.simulate_drug_kinetics(
                drug_params=drug_params,
                time_points=self.data_generator.time_points
            )
            traditional_predictions.append(pred)
        
        traditional_predictions = np.array(traditional_predictions)
        
        # Evaluate ensemble model
        ensemble_predictions = []
        for i in range(len(test_concentrations)):
            drug_props = self.drug_properties[-len(test_concentrations) + i]
            drug_params = self._convert_drug_properties(drug_props)
            pred, _ = self.ensemble_model.predict(
                drug_params=drug_params,
                time_points=self.data_generator.time_points
            )
            ensemble_predictions.append(pred)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Compute metrics
        metrics = self.compute_metrics(
            test_concentrations, 
            gnn_predictions, 
            traditional_predictions, 
            ensemble_predictions
        )
        
        self.results = {
            'metrics': metrics,
            'experiment_info': {
                'n_drugs': self.n_drugs,
                'n_epochs': self.n_epochs,
                'concentration_range': [float(self.concentrations.min()), float(self.concentrations.max())],
                'concentration_mean': float(self.concentrations.mean()),
                'dataset_shape': list(self.concentrations.shape),
                'timestamp': str(pd.Timestamp.now())
            }
        }
        
        return metrics
    
    def compute_metrics(self, true, gnn_pred, traditional_pred, ensemble_pred):
        """Compute evaluation metrics"""
        # Flatten data for metric computation
        true_flat = true.flatten()
        gnn_flat = gnn_pred.flatten()
        traditional_flat = traditional_pred.flatten()
        ensemble_flat = ensemble_pred.flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(true_flat) & np.isfinite(gnn_flat) & np.isfinite(traditional_flat) & np.isfinite(ensemble_flat)
        true_flat = true_flat[mask]
        gnn_flat = gnn_flat[mask]
        traditional_flat = traditional_flat[mask]
        ensemble_flat = ensemble_flat[mask]
        
        # Compute metrics
        def safe_mape(y_true, y_pred):
            """Safe MAPE computation avoiding division by zero"""
            mask = y_true > 1e-6  # Avoid division by very small numbers
            if np.sum(mask) == 0:
                return 100.0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        metrics = {
            'gnn': {
                'mape': safe_mape(true_flat, gnn_flat),
                'rmse': np.sqrt(mean_squared_error(true_flat, gnn_flat)),
                'r2': r2_score(true_flat, gnn_flat),
                'mae': np.mean(np.abs(true_flat - gnn_flat))
            },
            'traditional': {
                'mape': safe_mape(true_flat, traditional_flat),
                'rmse': np.sqrt(mean_squared_error(true_flat, traditional_flat)),
                'r2': r2_score(true_flat, traditional_flat),
                'mae': np.mean(np.abs(true_flat - traditional_flat))
            },
            'ensemble': {
                'mape': safe_mape(true_flat, ensemble_flat),
                'rmse': np.sqrt(mean_squared_error(true_flat, ensemble_flat)),
                'r2': r2_score(true_flat, ensemble_flat),
                'mae': np.mean(np.abs(true_flat - ensemble_flat))
            }
        }
        
        # Statistical significance testing
        gnn_errors = np.abs(true_flat - gnn_flat)
        traditional_errors = np.abs(true_flat - traditional_flat)
        
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(traditional_errors, gnn_errors)
        
        # Wilcoxon signed-rank test
        w_stat, w_pvalue = stats.wilcoxon(traditional_errors, gnn_errors)
        
        # MAPE improvement
        mape_improvement = metrics['traditional']['mape'] - metrics['gnn']['mape']
        mape_improvement_pct = (mape_improvement / metrics['traditional']['mape']) * 100
        
        metrics['statistical_tests'] = {
            'paired_t_test': {'statistic': float(t_stat), 'p_value': float(t_pvalue)},
            'wilcoxon_test': {'statistic': float(w_stat), 'p_value': float(w_pvalue)},
            'mape_improvement': float(mape_improvement),
            'mape_improvement_pct': float(mape_improvement_pct)
        }
        
        return metrics
    
    def save_results(self, filename: str = 'experiment_results.json'):
        """Save results to JSON file"""
        print(f"Saving results to {filename}...")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_to_save = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_results(self):
        """Print experiment results"""
        if not self.results:
            print("No results available. Run evaluate_models() first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        # Get dataset info from results or current data
        if 'experiment_info' in self.results:
            info = self.results['experiment_info']
            print(f"\nDataset: {info['n_drugs']} drugs, {info['dataset_shape'][1]} time points, {info['dataset_shape'][2]} organs")
            print(f"Concentration range: [{info['concentration_range'][0]:.3f}, {info['concentration_range'][1]:.3f}] mg/L")
        else:
            print(f"\nDataset: {self.n_drugs} drugs, {self.concentrations.shape[1]} time points, {self.concentrations.shape[2]} organs")
            print(f"Concentration range: [{self.concentrations.min():.3f}, {self.concentrations.max():.3f}] mg/L")
        
        print(f"\nModel Performance:")
        print(f"{'Model':<12} {'MAPE (%)':<10} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
        print("-" * 60)
        
        for model_name in ['traditional', 'ensemble', 'gnn']:
            m = metrics[model_name]
            print(f"{model_name.capitalize():<12} {m['mape']:<10.2f} {m['rmse']:<10.4f} {m['r2']:<10.4f} {m['mae']:<10.4f}")
        
        print(f"\nStatistical Significance:")
        stats = metrics['statistical_tests']
        print(f"MAPE Improvement: {stats['mape_improvement']:.2f}% ({stats['mape_improvement_pct']:.1f}% relative)")
        print(f"Paired t-test p-value: {stats['paired_t_test']['p_value']:.4f}")
        print(f"Wilcoxon test p-value: {stats['wilcoxon_test']['p_value']:.4f}")
        
        if stats['paired_t_test']['p_value'] < 0.05:
            print("✅ GNN significantly outperforms traditional PBPK (p < 0.05)")
        else:
            print("❌ No significant difference between GNN and traditional PBPK")
    
    def run_full_experiment(self):
        """Run the complete experiment"""
        results_path = 'experiment_results.json'
        
        # Check if results already exist and force_retrain is False
        if os.path.exists(results_path) and not self.force_retrain:
            print(f"📁 Found existing results at {results_path}")
            print("Loading existing results...")
            try:
                with open(results_path, 'r') as f:
                    self.results = json.load(f)
                print("✅ Successfully loaded existing results!")
                self.print_results()
                return self.results.get('metrics', {})
            except Exception as e:
                print(f"❌ Failed to load results: {e}")
                print("Running new experiment...")
        elif self.force_retrain:
            print("🔄 Force retrain enabled - running new experiment...")
        
        print("Starting full experiment...")
        print("="*60)
        
        # Generate data
        self.generate_data()
        
        # Prepare GNN data
        self.prepare_gnn_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train GNN model
        self.train_gnn_model()
        
        # Evaluate models
        metrics = self.evaluate_models()
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        return metrics

def main():
    """Main function"""
    import sys
    
    print("GNN-PBPK Experiment")
    print("="*60)
    
    # Check for command line arguments
    force_retrain = '--retrain' in sys.argv or '--force' in sys.argv
    
    if force_retrain:
        print("🔄 Force retrain mode enabled")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run experiment
    experiment = ExperimentRunner(n_drugs=30, n_epochs=100, force_retrain=force_retrain)
    results = experiment.run_full_experiment()
    
    return results

if __name__ == "__main__":
    results = main()
