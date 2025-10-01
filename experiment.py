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
from scipy import stats
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
        
        # Split data into exact 70%/15%/15%
        total_indices = np.arange(len(gnn_data))
        # First hold out 15% for test
        trainval_indices, test_indices = train_test_split(
            total_indices, test_size=0.15, random_state=self.random_seed, shuffle=True
        )
        # From remaining 85%, hold out 15% of total for val => 0.15/0.85
        val_frac_of_trainval = 0.15 / 0.85
        train_indices, val_indices = train_test_split(
            trainval_indices, test_size=val_frac_of_trainval, random_state=self.random_seed, shuffle=True
        )

        # Persist indices for metric computation
        self.train_indices = np.sort(train_indices)
        self.val_indices = np.sort(val_indices)
        self.test_indices = np.sort(test_indices)

        # Materialize datasets
        train_data = [gnn_data[i] for i in self.train_indices]
        val_data = [gnn_data[i] for i in self.val_indices]
        test_data = [gnn_data[i] for i in self.test_indices]
        
        # Create data loaders
        self.train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test (target 70/15/15)")
        
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
                    time_steps=batch.time_steps
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
        
        # Get test data using stored indices from the split (exact 15%)
        if not hasattr(self, 'test_indices'):
            raise RuntimeError('Test indices not found. Call prepare_gnn_data() before compute_metrics().')
        test_concentrations = self.concentrations[self.test_indices]
        test_traditional = self.traditional_predictions[self.test_indices]
        test_ensemble = self.ensemble_predictions[self.test_indices]
        
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
        per_drug_errors = {}
        
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

            # Per-drug MAE and MAPE for statistical testing
            n_drugs_test = predictions.shape[0]
            model_mae = []
            model_mape = []
            for d in range(n_drugs_test):
                true_d = test_concentrations[d].flatten()
                pred_d = predictions[d].flatten()
                mask_d = true_d > 0
                mae_d = np.mean(np.abs(true_d - pred_d))
                mape_d = mean_absolute_percentage_error(true_d[mask_d], pred_d[mask_d]) * 100 if mask_d.any() else np.nan
                model_mae.append(mae_d)
                model_mape.append(mape_d)
            per_drug_errors[model_name] = {
                'MAE': np.array(model_mae),
                'MAPE': np.array(model_mape)
            }
        
        self.metrics = metrics
        self.per_drug_errors = per_drug_errors
        return metrics

    def statistical_significance(self):
        """Run paired significance tests between GNN-PBPK and baselines on per-drug errors."""
        if not hasattr(self, 'per_drug_errors'):
            raise RuntimeError('Compute metrics before statistical tests.')

        results = {}
        gnn_mae = self.per_drug_errors['GNN-PBPK']['MAE']
        gnn_mape = self.per_drug_errors['GNN-PBPK']['MAPE']

        for baseline in ['Traditional PBPK', 'Ensemble PBPK']:
            base_mae = self.per_drug_errors[baseline]['MAE']
            base_mape = self.per_drug_errors[baseline]['MAPE']

            # Align and drop NaNs for MAPE
            valid_mask = ~np.isnan(gnn_mape) & ~np.isnan(base_mape)
            g_mape = gnn_mape[valid_mask]
            b_mape = base_mape[valid_mask]

            # Paired t-test and Wilcoxon on MAE and MAPE (lower is better)
            t_mae = stats.ttest_rel(base_mae, gnn_mae, alternative='greater')
            w_mae = stats.wilcoxon(base_mae, gnn_mae, alternative='greater')
            t_mape = stats.ttest_rel(b_mape, g_mape, alternative='greater') if len(g_mape) > 0 else None
            w_mape = stats.wilcoxon(b_mape, g_mape, alternative='greater') if len(g_mape) > 0 else None

            # Bootstrap CI for improvement in MAPE
            def bootstrap_ci(diff, n_boot=5000, alpha=0.05):
                if len(diff) == 0:
                    return (np.nan, np.nan, np.nan)
                rng = np.random.default_rng(self.random_seed)
                boots = []
                for _ in range(n_boot):
                    idx = rng.integers(0, len(diff), len(diff))
                    boots.append(np.mean(diff[idx]))
                boots = np.sort(boots)
                lower = boots[int((alpha/2) * n_boot)]
                upper = boots[int((1 - alpha/2) * n_boot) - 1]
                return (np.mean(diff), lower, upper)

            mape_improvement = (b_mape - g_mape)  # positive means GNN better
            mape_improve_mean, mape_ci_lo, mape_ci_hi = bootstrap_ci(mape_improvement)

            results[baseline] = {
                'paired_t_MAE_p': float(t_mae.pvalue),
                'wilcoxon_MAE_p': float(w_mae.pvalue),
                'paired_t_MAPE_p': float(t_mape.pvalue) if t_mape is not None else np.nan,
                'wilcoxon_MAPE_p': float(w_mape.pvalue) if w_mape is not None else np.nan,
                'MAPE_improvement_mean_pct': float(mape_improve_mean),
                'MAPE_improvement_CI95_pct': (float(mape_ci_lo), float(mape_ci_hi))
            }

        self.significance_results = results
        return results
    
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
        
        # Significance (if available)
        sig = getattr(self, 'significance_results', {})

        summary = {
            'overall_performance': overall_df,
            'organ_performance': organ_df,
            'improvement_percentage': improvement,
            'gnn_vs_traditional': {
                'GNN MAPE': gnn_mape,
                'Traditional MAPE': trad_mape,
                'Improvement': improvement
            },
            'significance': sig
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
        # Step 6b: Statistical significance
        self.statistical_significance()
        
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
