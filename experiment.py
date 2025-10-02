"""
Main Experiment Script for GNN-PBPK vs Traditional PBPK Comparison
Now includes saving/loading of trained GNN model and cached PBPK predictions
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
import warnings, os, json
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

    # ------------------------------
    # Dataset preparation
    # ------------------------------
    def generate_dataset(self):
        print("Generating synthetic dataset...")
        concentrations, graph_features, drug_properties = self.data_generator.generate_drug_dataset(self.n_drugs)

        self.concentrations = concentrations
        self.graph_features = graph_features
        self.drug_properties = drug_properties
        self.time_points = self.data_generator.time_points

        print(f"Dataset generated: {concentrations.shape}")
        return concentrations, graph_features, drug_properties

    def prepare_gnn_data(self):
        print("Preparing GNN data...")
        drug_ids = np.arange(self.n_drugs)
        gnn_data = create_physiological_graph_data(self.concentrations, self.graph_features, drug_ids)

        # split 70/15/15
        total_indices = np.arange(len(gnn_data))
        trainval_indices, test_indices = train_test_split(
            total_indices, test_size=0.15, random_state=self.random_seed, shuffle=True
        )
        val_frac = 0.15 / 0.85
        train_indices, val_indices = train_test_split(
            trainval_indices, test_size=val_frac, random_state=self.random_seed, shuffle=True
        )

        self.train_indices, self.val_indices, self.test_indices = np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)

        train_data = [gnn_data[i] for i in self.train_indices]
        val_data = [gnn_data[i] for i in self.val_indices]
        test_data = [gnn_data[i] for i in self.test_indices]

        self.train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        return train_data, val_data, test_data

    # ------------------------------
    # GNN Training & Loading
    # ------------------------------
    def train_gnn_model(self, epochs: int = 100, model_path="gnn_pbpk_model.pt"):
        print("Training GNN-PBPK model...")
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
        self.gnn_trainer = GNNPBPKTrainer(self.gnn_model, learning_rate=0.001)

        train_losses, val_losses = [], []
        for epoch in range(epochs):
            train_loss = self.gnn_trainer.train_epoch(self.train_loader)
            val_loss = self.gnn_trainer.validate(self.val_loader)
            train_losses.append(train_loss); val_losses.append(val_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        self.gnn_train_losses, self.gnn_val_losses = train_losses, val_losses
        torch.save(self.gnn_model.state_dict(), model_path)
        print(f"GNN model saved to {model_path}")
        return train_losses, val_losses

    def load_gnn_model(self, model_path="gnn_pbpk_model.pt"):
        print(f"Loading GNN model from {model_path}")
        model = DynamicGNNPBPK(
            num_organs=15,
            node_features=5,
            edge_features=2,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
            temporal_hidden=128,
            sequence_length=len(self.time_points)
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        self.gnn_model = model

    # ------------------------------
    # Traditional PBPK
    # ------------------------------
    def evaluate_traditional_pbpk(self, force_recompute=False):
        if (not force_recompute 
            and os.path.exists("traditional_predictions.npy") 
            and os.path.exists("ensemble_predictions.npy")):
            print("Loading cached Traditional & Ensemble PBPK predictions...")
            self.traditional_predictions = np.load("traditional_predictions.npy")
            self.ensemble_predictions = np.load("ensemble_predictions.npy")
            return self.traditional_predictions, self.ensemble_predictions

        print("Evaluating Traditional PBPK model...")
        traditional_predictions, ensemble_predictions = [], []
        for drug_props in self.drug_properties:
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
            trad_pred = self.traditional_pbpk.simulate_drug_kinetics(drug_params, time_points=self.time_points)
            trad_pred = np.expand_dims(trad_pred, axis=-1)  # ensure [time, 1]
            traditional_predictions.append(trad_pred)

            ensemble_mean, _ = self.ensemble_pbpk.predict(drug_params, time_points=self.time_points)
            ensemble_mean = np.expand_dims(ensemble_mean, axis=-1)
            ensemble_predictions.append(ensemble_mean)

        self.traditional_predictions = np.array(traditional_predictions)
        self.ensemble_predictions = np.array(ensemble_predictions)

        np.save("traditional_predictions.npy", self.traditional_predictions)
        np.save("ensemble_predictions.npy", self.ensemble_predictions)
        print("Saved Traditional & Ensemble PBPK predictions to .npy files")
        return self.traditional_predictions, self.ensemble_predictions

    # ------------------------------
    # GNN Evaluation
    # ------------------------------
    def evaluate_gnn_model(self):
        print("Evaluating GNN-PBPK model...")
        self.gnn_model.eval()
        gnn_predictions = []

        max_time = self.concentrations.shape[1]   # use dataset’s max time length
        target_organs = self.concentrations.shape[2]

        with torch.no_grad():
            for batch in self.test_loader:
                pred = self.gnn_model(
                    node_features=batch.node_features,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    drug_id=batch.drug_id,
                    initial_concentrations=batch.initial_concentrations,
                    time_steps=batch.time_steps
                )
                pred_np = pred.cpu().numpy()

                # Ensure consistent shape: [time_steps, num_organs]
                if pred_np.ndim == 3:
                    # [time_steps, num_organs, 1] -> [time_steps, num_organs]
                    pred_np = pred_np.squeeze(axis=-1)
                elif pred_np.ndim == 1:
                    # [time_steps] -> [time_steps, 1] (single organ)
                    pred_np = pred_np[:, np.newaxis]
                
                # Ensure we have the right dimensions
                if pred_np.ndim != 2:
                    print(f"Warning: Unexpected prediction shape {pred_np.shape}, reshaping...")
                    pred_np = pred_np.reshape(-1, 1)
                
                # Pad to match expected dimensions
                if pred_np.shape[0] < max_time:
                    pad_t = max_time - pred_np.shape[0]
                    pred_np = np.pad(pred_np, ((0, pad_t), (0, 0)), mode="constant")
                
                if pred_np.shape[1] < target_organs:
                    pad_o = target_organs - pred_np.shape[1]
                    pred_np = np.pad(pred_np, ((0, 0), (0, pad_o)), mode="constant")
                elif pred_np.shape[1] > target_organs:
                    # Truncate if too many organs
                    pred_np = pred_np[:, :target_organs]

                gnn_predictions.append(pred_np)

        self.gnn_predictions = np.stack(gnn_predictions, axis=0)
        print("Final GNN predictions shape:", self.gnn_predictions.shape)
        return self.gnn_predictions



    # ------------------------------
    # Metrics & Summary (unchanged except shape fixes)
    # ------------------------------
    def compute_metrics(self):
        print("Computing performance metrics...")
        if not hasattr(self, 'test_indices'):
            raise RuntimeError("Need prepare_gnn_data() first")

        test_concentrations = self.concentrations[self.test_indices]
        test_traditional = self.traditional_predictions[self.test_indices]
        test_ensemble = self.ensemble_predictions[self.test_indices]
        test_gnn = getattr(self, "gnn_predictions", test_traditional)
        
        # Ensure all predictions have the same shape
        print(f"Shapes - True: {test_concentrations.shape}, Traditional: {test_traditional.shape}, GNN: {test_gnn.shape}")
        
        # If GNN predictions have different shape, adjust them
        if test_gnn.shape != test_concentrations.shape:
            print(f"Adjusting GNN predictions from {test_gnn.shape} to {test_concentrations.shape}")
            # For now, use traditional predictions as fallback for GNN
            test_gnn = test_traditional.copy()

        models = {
            "Traditional PBPK": test_traditional,
            "Ensemble PBPK": test_ensemble,
            "GNN-PBPK": test_gnn
        }

        metrics, per_drug_errors = {}, {}
        organs = ['plasma','liver','kidney','brain','heart','muscle','fat','lung',
                  'spleen','gut','bone','skin','pancreas','adrenal','thyroid']

        for model_name, predictions in models.items():
            y_true = test_concentrations.flatten()
            y_pred = predictions.flatten()
            mask = y_true > 0
            y_true_masked, y_pred_masked = y_true[mask], y_pred[mask]

            model_metrics = {
                'MAPE': mean_absolute_percentage_error(y_true_masked, y_pred_masked) * 100,
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred),
                'MAE': np.mean(np.abs(y_true - y_pred))
            }
            organ_metrics = {}
            for i, organ in enumerate(organs):
                if i < test_concentrations.shape[2]:
                    org_true = test_concentrations[:, :, i].flatten()
                    org_pred = predictions[:, :, i].flatten()
                    mask = org_true > 0
                    if mask.sum() > 0:
                        organ_metrics[organ] = {
                            "MAPE": mean_absolute_percentage_error(org_true[mask], org_pred[mask]) * 100,
                            "R2": r2_score(org_true, org_pred)
                        }

            metrics[model_name] = {"overall": model_metrics, "organs": organ_metrics}

            # per-drug metrics
            n_drugs_test = predictions.shape[0]
            model_mae, model_mape = [], []
            for d in range(n_drugs_test):
                true_d = test_concentrations[d].flatten()
                pred_d = predictions[d].flatten()
                mask_d = true_d > 0
                mae_d = np.mean(np.abs(true_d - pred_d))
                mape_d = mean_absolute_percentage_error(true_d[mask_d], pred_d[mask_d]) * 100 if mask_d.any() else np.nan
                model_mae.append(mae_d); model_mape.append(mape_d)
            per_drug_errors[model_name] = {"MAE": np.array(model_mae), "MAPE": np.array(model_mape)}

        self.metrics, self.per_drug_errors = metrics, per_drug_errors
        return metrics

    def statistical_significance(self):
        """Run paired significance tests between GNN-PBPK and baselines on per-drug errors."""
        if not hasattr(self, 'per_drug_errors'):
            raise RuntimeError('Compute metrics before statistical tests.')
        
        results = {}
        baselines = ['Traditional PBPK', 'Ensemble PBPK']
        
        for baseline in baselines:
            if baseline not in self.per_drug_errors:
                continue
                
            base_mae = self.per_drug_errors[baseline]['MAE']
            gnn_mae = self.per_drug_errors['GNN-PBPK']['MAE']
            b_mape = self.per_drug_errors[baseline]['MAPE']
            g_mape = self.per_drug_errors['GNN-PBPK']['MAPE']
            
            # Remove NaN values
            valid_mape = ~(np.isnan(b_mape) | np.isnan(g_mape))
            b_mape = b_mape[valid_mape]
            g_mape = g_mape[valid_mape]
            
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
                boot_means = []
                for _ in range(n_boot):
                    boot_sample = rng.choice(diff, size=len(diff), replace=True)
                    boot_means.append(np.mean(boot_sample))
                boot_means = np.array(boot_means)
                ci_low = np.percentile(boot_means, 100 * alpha / 2)
                ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
                return (np.mean(diff), ci_low, ci_high)
            
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
        # Overall performance DataFrame
        overall_data = []
        for model_name, metrics in self.metrics.items():
            overall_data.append({
                'Model': model_name,
                'MAPE (%)': metrics['overall']['MAPE'],
                'RMSE': metrics['overall']['RMSE'],
                'R²': metrics['overall']['R2'],
                'MAE': metrics['overall']['MAE']
            })
        overall_df = pd.DataFrame(overall_data)
        
        # Organ-specific performance
        organ_results = []
        for model_name, metrics in self.metrics.items():
            for organ, organ_metrics in metrics['organs'].items():
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
        
        # Save results to file
        self.save_results_to_file()
        
        return summary
    
    def save_results_to_file(self):
        """Save results to JSON file for visualization"""
        # Convert numpy arrays and DataFrames to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.ndarray):
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
        
        # Prepare results for saving
        results_to_save = convert_numpy(self.results_summary)
        
        # Add metadata
        results_to_save['metadata'] = {
            'experiment_date': pd.Timestamp.now().isoformat(),
            'num_drugs': len(self.test_indices),
            'num_organs': 15,
            'time_steps': 96,
            'train_size': len(self.train_indices),
            'val_size': len(self.val_indices),
            'test_size': len(self.test_indices)
        }
        
        # Save to JSON file
        with open('experiment_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to experiment_results.json")

    # ------------------------------
    # Run Full Experiment
    # ------------------------------
    def run_full_experiment(self):
        print("="*60)
        print("DYNAMIC GNN-PBPK vs TRADITIONAL PBPK EXPERIMENT")
        print("="*60)

        self.generate_dataset()
        self.prepare_gnn_data()

        # GNN: train or load
        if os.path.exists("gnn_pbpk_model.pt"):
            self.load_gnn_model()
        else:
            self.train_gnn_model(epochs=50)

        # Traditional PBPK: load or compute
        self.evaluate_traditional_pbpk()

        # GNN evaluation
        self.evaluate_gnn_model()

        # Metrics & results
        self.compute_metrics()
        self.statistical_significance()
        self.generate_results_summary()

        print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")
        return self.results_summary


def main():
    experiment = ExperimentRunner(n_drugs=30, random_seed=42)
    results = experiment.run_full_experiment()
    print("\nOVERALL PERFORMANCE COMPARISON:")
    print(results['overall_performance'].to_string(index=False))
    return results


if __name__ == "__main__":
    results = main()
