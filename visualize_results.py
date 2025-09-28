"""
Visualization and Analysis Script for GNN-PBPK Experiment Results
Creates comprehensive plots and analysis of model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn')
sns.set_palette("husl")

class ResultsVisualizer:
    """
    Creates comprehensive visualizations for GNN-PBPK experiment results
    """
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 300
        
    def create_performance_comparison_plot(self, results_data):
        """Create overall performance comparison plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        models = results_data['overall_performance']['Model'].values
        mape = results_data['overall_performance']['MAPE (%)'].values
        rmse = results_data['overall_performance']['RMSE'].values
        r2 = results_data['overall_performance']['R²'].values
        mae = results_data['overall_performance']['MAE'].values
        
        # MAPE comparison
        bars1 = ax1.bar(models, mape, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('MAPE (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mape):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # RMSE comparison
        bars2 = ax2.bar(models, rmse, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, rmse):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # R² comparison
        bars3 = ax3.bar(models, r2, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('R² Correlation Coefficient', fontsize=14, fontweight='bold')
        ax3.set_ylabel('R²')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars3, r2):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE comparison
        bars4 = ax4.bar(models, mae, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('MAE')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, mae):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_organ_performance_heatmap(self, results_data):
        """Create organ-specific performance heatmap"""
        
        # Pivot data for heatmap
        organ_df = results_data['organ_performance']
        mape_pivot = organ_df.pivot(index='Organ', columns='Model', values='MAPE (%)')
        r2_pivot = organ_df.pivot(index='Organ', columns='Model', values='R²')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # MAPE heatmap
        sns.heatmap(mape_pivot, annot=True, fmt='.1f', cmap='Reds_r', 
                   ax=ax1, cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title('Organ-Specific MAPE Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Organ')
        
        # R² heatmap
        sns.heatmap(r2_pivot, annot=True, fmt='.3f', cmap='Greens', 
                   ax=ax2, cbar_kws={'label': 'R²'})
        ax2.set_title('Organ-Specific R² Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Organ')
        
        plt.tight_layout()
        plt.savefig('organ_performance_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_improvement_analysis(self, results_data):
        """Create improvement analysis visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall improvement
        improvement = results_data['improvement_percentage']
        
        # Create improvement bar chart
        models = ['Traditional PBPK', 'GNN-PBPK']
        mape_values = [results_data['gnn_vs_traditional']['Traditional MAPE'],
                      results_data['gnn_vs_traditional']['GNN MAPE']]
        
        bars = ax1.bar(models, mape_values, color=['#FF6B6B', '#45B7D1'])
        ax1.set_title('GNN-PBPK vs Traditional PBPK\nOverall Performance Improvement', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('MAPE (%)')
        
        # Add value labels and improvement percentage
        for bar, value in zip(bars, mape_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement arrow
        ax1.annotate(f'{improvement:.1f}% Improvement', 
                    xy=(0.5, max(mape_values)/2), xytext=(0.5, max(mape_values)/2 + 5),
                    ha='center', fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        # Organ-specific improvements
        organ_df = results_data['organ_performance']
        organ_improvements = []
        organs = []
        
        for organ in organ_df['Organ'].unique():
            organ_data = organ_df[organ_df['Organ'] == organ]
            trad_mape = organ_data[organ_data['Model'] == 'Traditional PBPK']['MAPE (%)'].iloc[0]
            gnn_mape = organ_data[organ_data['Model'] == 'GNN-PBPK']['MAPE (%)'].iloc[0]
            improvement = ((trad_mape - gnn_mape) / trad_mape) * 100
            organ_improvements.append(improvement)
            organs.append(organ)
        
        # Sort by improvement
        sorted_data = sorted(zip(organs, organ_improvements), key=lambda x: x[1], reverse=True)
        sorted_organs, sorted_improvements = zip(*sorted_data)
        
        colors = ['green' if x > 0 else 'red' for x in sorted_improvements]
        bars2 = ax2.barh(range(len(sorted_organs)), sorted_improvements, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_organs)))
        ax2.set_yticklabels(sorted_organs)
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Organ-Specific Performance Improvements\n(GNN-PBPK vs Traditional PBPK)', 
                     fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, sorted_improvements)):
            ax2.text(value + (1 if value > 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left' if value > 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('improvement_analysis.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_concentration_profiles(self, sample_data):
        """Create sample concentration-time profiles"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        organs = ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        models = ['Traditional PBPK', 'Ensemble PBPK', 'GNN-PBPK']
        
        for i, organ in enumerate(organs):
            ax = axes[i]
            
            # Plot concentration profiles for each model
            for j, (model, color) in enumerate(zip(models, colors)):
                if model in sample_data:
                    concentrations = sample_data[model][:, i]  # Assuming first drug
                    time_points = np.arange(len(concentrations)) * 0.5  # 0.5h intervals
                    ax.plot(time_points, concentrations, color=color, linewidth=2, 
                           label=model, alpha=0.8)
            
            ax.set_title(f'{organ.capitalize()} Concentration Profile', fontweight='bold')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Concentration (mg/L)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('concentration_profiles.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_model_architecture_diagram(self):
        """Create GNN-PBPK model architecture diagram"""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define positions for components
        positions = {
            'Input': (1, 8),
            'Graph Construction': (3, 8),
            'GAT Layers': (5, 8),
            'Message Passing': (7, 8),
            'Temporal LSTM': (9, 8),
            'Output': (11, 8),
            'Drug Embedding': (5, 6),
            'Node Features': (3, 6),
            'Edge Features': (3, 4)
        }
        
        # Draw components
        components = {
            'Input': {'size': (1.5, 0.8), 'color': '#E8F4FD'},
            'Graph Construction': {'size': (1.5, 0.8), 'color': '#FFF2CC'},
            'GAT Layers': {'size': (1.5, 0.8), 'color': '#D5E8D4'},
            'Message Passing': {'size': (1.5, 0.8), 'color': '#F8CECC'},
            'Temporal LSTM': {'size': (1.5, 0.8), 'color': '#E1D5E7'},
            'Output': {'size': (1.5, 0.8), 'color': '#FFE6CC'},
            'Drug Embedding': {'size': (1.2, 0.6), 'color': '#F0F0F0'},
            'Node Features': {'size': (1.2, 0.6), 'color': '#F0F0F0'},
            'Edge Features': {'size': (1.2, 0.6), 'color': '#F0F0F0'}
        }
        
        for name, pos in positions.items():
            size = components[name]['size']
            color = components[name]['color']
            
            rect = Rectangle((pos[0] - size[0]/2, pos[1] - size[1]/2), 
                           size[0], size[1], facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], name, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw arrows
        arrows = [
            ('Input', 'Graph Construction'),
            ('Graph Construction', 'GAT Layers'),
            ('GAT Layers', 'Message Passing'),
            ('Message Passing', 'Temporal LSTM'),
            ('Temporal LSTM', 'Output'),
            ('Drug Embedding', 'GAT Layers'),
            ('Node Features', 'Graph Construction'),
            ('Edge Features', 'Graph Construction')
        ]
        
        for start, end in arrows:
            start_pos = positions[start]
            end_pos = positions[end]
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add title and description
        ax.set_title('Dynamic GNN-PBPK Model Architecture', fontsize=16, fontweight='bold', pad=20)
        ax.text(6, 2, 'The model learns temporal evolution of drug concentrations\nthrough graph-based message passing between physiological organs', 
               ha='center', va='center', fontsize=12, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(1, 10)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('gnn_architecture.png', dpi=self.dpi, bbox_inches='tight')
        plt.show()

def generate_synthetic_results():
    """Generate synthetic results for demonstration"""
    
    # Create synthetic results data
    overall_performance = pd.DataFrame({
        'Model': ['Traditional PBPK', 'Ensemble PBPK', 'GNN-PBPK'],
        'MAPE (%)': [25.3, 22.1, 18.7],
        'RMSE': [0.156, 0.142, 0.128],
        'R²': [0.847, 0.872, 0.901],
        'MAE': [0.089, 0.081, 0.073]
    })
    
    # Organ-specific performance
    organs = ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 'fat', 'lung']
    organ_data = []
    
    for organ in organs:
        for model in ['Traditional PBPK', 'Ensemble PBPK', 'GNN-PBPK']:
            # Generate realistic performance metrics
            base_mape = np.random.uniform(15, 35)
            base_r2 = np.random.uniform(0.7, 0.95)
            
            if model == 'GNN-PBPK':
                mape = base_mape * 0.8  # 20% improvement
                r2 = min(0.99, base_r2 * 1.05)  # 5% improvement
            elif model == 'Ensemble PBPK':
                mape = base_mape * 0.9  # 10% improvement
                r2 = min(0.99, base_r2 * 1.02)  # 2% improvement
            else:
                mape = base_mape
                r2 = base_r2
            
            organ_data.append({
                'Model': model,
                'Organ': organ,
                'MAPE (%)': mape,
                'R²': r2
            })
    
    organ_performance = pd.DataFrame(organ_data)
    
    # Improvement analysis
    improvement_percentage = 26.1  # 26.1% improvement
    
    results_data = {
        'overall_performance': overall_performance,
        'organ_performance': organ_performance,
        'improvement_percentage': improvement_percentage,
        'gnn_vs_traditional': {
            'GNN MAPE': 18.7,
            'Traditional MAPE': 25.3,
            'Improvement': improvement_percentage
        }
    }
    
    return results_data

def main():
    """Generate all visualizations"""
    
    print("Generating GNN-PBPK Experiment Visualizations...")
    
    # Generate synthetic results
    results_data = generate_synthetic_results()
    
    # Create visualizer
    visualizer = ResultsVisualizer()
    
    # Generate all plots
    print("1. Creating performance comparison plot...")
    visualizer.create_performance_comparison_plot(results_data)
    
    print("2. Creating organ performance heatmap...")
    visualizer.create_organ_performance_heatmap(results_data)
    
    print("3. Creating improvement analysis...")
    visualizer.create_improvement_analysis(results_data)
    
    print("4. Creating model architecture diagram...")
    visualizer.create_model_architecture_diagram()
    
    # Create sample concentration profiles
    print("5. Creating sample concentration profiles...")
    time_decay = np.exp(-np.arange(96) * 0.02).reshape(-1, 1)
    sample_data = {
        'Traditional PBPK': np.random.exponential(0.5, (96, 6)) * time_decay,
        'Ensemble PBPK': np.random.exponential(0.45, (96, 6)) * time_decay,
        'GNN-PBPK': np.random.exponential(0.4, (96, 6)) * time_decay
    }
    visualizer.create_concentration_profiles(sample_data)
    
    print("\nAll visualizations generated successfully!")
    print("Files saved:")
    print("- performance_comparison.png")
    print("- organ_performance_heatmap.png") 
    print("- improvement_analysis.png")
    print("- gnn_architecture.png")
    print("- concentration_profiles.png")
    
    return results_data

if __name__ == "__main__":
    results = main()
