"""
Synthetic Dataset Generator for GNN-PBPK Model
Generates realistic preclinical drug concentration data for multiple tissues
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Dict, List, Tuple
import random

class PhysiologicalGraph:
    """Represents the physiological graph structure for PBPK modeling"""
    
    def __init__(self):
        self.organs = [
            'plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 
            'fat', 'lung', 'spleen', 'gut', 'bone', 'skin', 'pancreas',
            'adrenal', 'thyroid'
        ]
        
        # Organ volumes (L/kg body weight)
        self.volumes = {
            'plasma': 0.055, 'liver': 0.026, 'kidney': 0.004, 'brain': 0.02,
            'heart': 0.004, 'muscle': 0.4, 'fat': 0.19, 'lung': 0.007,
            'spleen': 0.002, 'gut': 0.017, 'bone': 0.14, 'skin': 0.11,
            'pancreas': 0.001, 'adrenal': 0.0002, 'thyroid': 0.0002
        }
        
        # Blood flow rates (L/min/kg)
        self.blood_flows = {
            'plasma': 5.0, 'liver': 0.8, 'kidney': 0.6, 'brain': 0.5,
            'heart': 0.3, 'muscle': 1.2, 'fat': 0.2, 'lung': 5.0,
            'spleen': 0.1, 'gut': 0.4, 'bone': 0.1, 'skin': 0.3,
            'pancreas': 0.05, 'adrenal': 0.01, 'thyroid': 0.01
        }
        
        # Create adjacency matrix based on physiological connections
        self.adjacency_matrix = self._create_adjacency_matrix()
        
    def _create_adjacency_matrix(self) -> np.ndarray:
        """Create adjacency matrix based on physiological blood flow connections"""
        n_organs = len(self.organs)
        adj = np.zeros((n_organs, n_organs))
        
        # Define connections (from -> to)
        connections = [
            ('plasma', 'liver'), ('plasma', 'kidney'), ('plasma', 'brain'),
            ('plasma', 'heart'), ('plasma', 'muscle'), ('plasma', 'fat'),
            ('plasma', 'lung'), ('plasma', 'spleen'), ('plasma', 'gut'),
            ('plasma', 'bone'), ('plasma', 'skin'), ('plasma', 'pancreas'),
            ('plasma', 'adrenal'), ('plasma', 'thyroid'),
            ('liver', 'plasma'), ('kidney', 'plasma'), ('brain', 'plasma'),
            ('heart', 'plasma'), ('muscle', 'plasma'), ('fat', 'plasma'),
            ('lung', 'plasma'), ('spleen', 'plasma'), ('gut', 'plasma'),
            ('bone', 'plasma'), ('skin', 'plasma'), ('pancreas', 'plasma'),
            ('adrenal', 'plasma'), ('thyroid', 'plasma')
        ]
        
        for from_organ, to_organ in connections:
            from_idx = self.organs.index(from_organ)
            to_idx = self.organs.index(to_organ)
            adj[from_idx, to_idx] = 1
            
        return adj

class DrugProperties:
    """Represents drug-specific properties for PBPK modeling"""
    
    def __init__(self, drug_id: int):
        self.drug_id = drug_id
        
        # Generate realistic drug properties
        self.molecular_weight = np.random.uniform(150, 600)  # Da (realistic range)
        self.log_p = np.random.uniform(-1, 5)  # Lipophilicity
        self.fu_plasma = np.random.uniform(0.01, 0.95)  # Unbound fraction in plasma
        self.clearance = np.random.uniform(0.001, 0.033)  # L/min/kg (0.06-2.0 L/h/kg)
        self.volume_distribution = np.random.uniform(0.1, 20.0)  # L/kg
        
        # Tissue-specific properties
        self.tissue_affinity = self._generate_tissue_affinity()
        self.metabolic_rate = self._generate_metabolic_rate()
        self.transporter_mediated = np.random.choice([True, False], p=[0.3, 0.7])
        
    def _generate_tissue_affinity(self) -> Dict[str, float]:
        """Generate tissue-specific affinity coefficients"""
        affinity = {}
        for organ in ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 'fat', 
                     'lung', 'spleen', 'gut', 'bone', 'skin', 'pancreas',
                     'adrenal', 'thyroid']:
            if organ == 'plasma':  # Plasma has unit affinity
                affinity[organ] = 1.0
            # Higher affinity for lipophilic tissues (fat, brain) for lipophilic drugs
            elif organ in ['fat', 'brain'] and self.log_p > 2:
                affinity[organ] = np.random.uniform(2.0, 5.0)
            elif organ in ['liver', 'kidney']:  # High metabolic/elimination organs
                affinity[organ] = np.random.uniform(1.5, 3.0)
            else:
                affinity[organ] = np.random.uniform(0.5, 2.0)
        return affinity
    
    def _generate_metabolic_rate(self) -> Dict[str, float]:
        """Generate organ-specific metabolic rates"""
        metabolic_rate = {}
        for organ in ['plasma', 'liver', 'kidney', 'brain', 'heart', 'muscle', 'fat', 
                     'lung', 'spleen', 'gut', 'bone', 'skin', 'pancreas',
                     'adrenal', 'thyroid']:
            if organ == 'plasma':  # No metabolism in plasma
                metabolic_rate[organ] = 0.0
            elif organ == 'liver':  # Primary metabolic organ (L/min/kg)
                metabolic_rate[organ] = np.random.uniform(0.001, 0.017)  # 0.06-1.0 L/h/kg
            elif organ == 'kidney':  # Secondary elimination (L/min/kg)
                metabolic_rate[organ] = np.random.uniform(0.0002, 0.005)  # 0.012-0.3 L/h/kg
            else:
                metabolic_rate[organ] = np.random.uniform(0.00002, 0.0008)  # 0.001-0.05 L/h/kg
        return metabolic_rate

class SyntheticDataGenerator:
    """Generates synthetic preclinical drug concentration data"""
    
    def __init__(self):
        self.graph = PhysiologicalGraph()
        self.time_points = np.arange(0, 48, 0.5)  # 48 hours, 0.5h intervals
        
    def generate_drug_dataset(self, n_drugs: int = 50) -> Tuple[np.ndarray, np.ndarray, List[DrugProperties]]:
        """Generate synthetic dataset for multiple drugs"""
        
        all_concentrations = []
        all_graph_features = []
        drug_properties = []
        
        for drug_id in range(n_drugs):
            drug_props = DrugProperties(drug_id)
            drug_properties.append(drug_props)
            
            # Generate concentration-time profiles
            concentrations = self._simulate_drug_kinetics(drug_props)
            all_concentrations.append(concentrations)
            
            # Generate graph features
            graph_features = self._generate_graph_features(drug_props)
            all_graph_features.append(graph_features)
            
        return (np.array(all_concentrations), 
                np.array(all_graph_features), 
                drug_properties)
    
    def _simulate_drug_kinetics(self, drug_props: DrugProperties) -> np.ndarray:
        """Simulate drug concentration-time profiles using simplified PBPK model"""
        
        def pbpk_ode(y, t, drug_props, graph):
            """ODE system for PBPK model - fixed and stable"""
            concentrations = np.maximum(y, 0.0)  # Ensure non-negative
            dydt = np.zeros_like(concentrations)
            
            for i, organ in enumerate(graph.organs):
                if organ == 'plasma':
                    # Plasma: receives from all organs, distributes to all organs
                    total_inflow = 0
                    total_outflow = 0
                    
                    for j, other_organ in enumerate(graph.organs):
                        if j != i:  # Not plasma itself
                            # Flow from other organs to plasma
                            flow_rate = graph.blood_flows[other_organ] * 0.01  # Reduced flow rate
                            total_inflow += flow_rate * concentrations[j] * drug_props.fu_plasma
                            
                            # Flow from plasma to other organs
                            total_outflow += flow_rate * concentrations[i] * drug_props.fu_plasma
                    
                    # Clearance from plasma (realistic rate)
                    clearance = drug_props.clearance * concentrations[i] * 0.1  # Reduced clearance rate
                    
                    dydt[i] = (total_inflow - total_outflow - clearance) / graph.volumes[organ]
                    
                else:
                    # Other organs: exchange with plasma
                    plasma_idx = 0
                    flow_rate = graph.blood_flows[organ] * 0.01  # Reduced flow rate
                    
                    # Flow from plasma to organ
                    inflow = flow_rate * concentrations[plasma_idx] * drug_props.fu_plasma
                    
                    # Flow from organ to plasma
                    outflow = flow_rate * concentrations[i] * drug_props.fu_plasma
                    
                    # Tissue-specific metabolism (reduced)
                    metabolism = drug_props.metabolic_rate[organ] * concentrations[i] * 0.01
                    
                    dydt[i] = (inflow - outflow - metabolism) / graph.volumes[organ]
            
            return dydt
        
        # Initial conditions (dose in plasma)
        initial_dose = 0.1  # mg/kg (realistic therapeutic dose)
        y0 = np.zeros(len(self.graph.organs))
        y0[0] = initial_dose / self.graph.volumes['plasma']  # Initial plasma concentration
        
        # Solve ODE system with better numerical stability
        try:
            solution = odeint(pbpk_ode, y0, self.time_points, args=(drug_props, self.graph), 
                            rtol=1e-6, atol=1e-8, mxstep=10000)
        except:
            # Fallback with simpler solver settings
            solution = odeint(pbpk_ode, y0, self.time_points, args=(drug_props, self.graph), 
                            rtol=1e-3, atol=1e-5)
        
        # Ensure non-negative concentrations (physically realistic)
        solution = np.maximum(solution, 0.0)
        
        # Add realistic noise (5% of mean concentration)
        if np.any(solution > 0):
            noise_level = 0.05  # 5% noise
            mean_conc = np.mean(solution[solution > 0])
            noise = np.random.normal(0, noise_level * mean_conc, solution.shape)
            solution = np.maximum(solution + noise, 0.0)
        
        return solution
    
    def _generate_graph_features(self, drug_props: DrugProperties) -> np.ndarray:
        """Generate node features for the physiological graph"""
        n_organs = len(self.graph.organs)
        
        # Node features: [volume, blood_flow, tissue_affinity, metabolic_rate, fu_plasma]
        node_features = np.zeros((n_organs, 5))
        
        for i, organ in enumerate(self.graph.organs):
            node_features[i, 0] = self.graph.volumes[organ]
            node_features[i, 1] = self.graph.blood_flows[organ]
            node_features[i, 2] = drug_props.tissue_affinity.get(organ, 1.0)
            node_features[i, 3] = drug_props.metabolic_rate.get(organ, 0.01)
            node_features[i, 4] = drug_props.fu_plasma
        
        # Return node features as 2D array (n_organs, 5)
        return node_features

def main():
    """Generate and save synthetic dataset"""
    generator = SyntheticDataGenerator()
    
    print("Generating synthetic preclinical dataset...")
    concentrations, graph_features, drug_properties = generator.generate_drug_dataset(n_drugs=50)
    
    print(f"Dataset shape: {concentrations.shape}")
    print(f"Graph features shape: {graph_features.shape}")
    print(f"Number of drugs: {len(drug_properties)}")
    print(f"Number of organs: {len(generator.graph.organs)}")
    print(f"Time points: {len(generator.time_points)}")
    
    # Save dataset
    np.save('synthetic_concentrations.npy', concentrations)
    np.save('synthetic_graph_features.npy', graph_features)
    np.save('time_points.npy', generator.time_points)
    
    # Save drug properties
    drug_data = []
    for i, drug_props in enumerate(drug_properties):
        drug_data.append({
            'drug_id': drug_props.drug_id,
            'molecular_weight': drug_props.molecular_weight,
            'log_p': drug_props.log_p,
            'fu_plasma': drug_props.fu_plasma,
            'clearance': drug_props.clearance,
            'volume_distribution': drug_props.volume_distribution,
            'transporter_mediated': drug_props.transporter_mediated
        })
    
    drug_df = pd.DataFrame(drug_data)
    drug_df.to_csv('drug_properties.csv', index=False)
    
    print("Dataset saved successfully!")
    print(f"Concentration data: {concentrations.shape}")
    print(f"Graph features: {graph_features.shape}")
    print(f"Drug properties: {len(drug_data)} drugs")

if __name__ == "__main__":
    main()
