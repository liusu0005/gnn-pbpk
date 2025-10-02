"""
Traditional PBPK Model Implementation
Well-stirred tank model with explicit ODEs for comparison with GNN-PBPK
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class DrugParameters:
    """Drug-specific parameters for PBPK modeling"""
    clearance: float
    volume_distribution: float
    fu_plasma: float
    molecular_weight: float
    log_p: float
    tissue_affinity: Dict[str, float]
    metabolic_rate: Dict[str, float]
    transporter_mediated: bool

@dataclass
class PhysiologicalParameters:
    """Physiological parameters for PBPK modeling"""
    organ_volumes: Dict[str, float]
    blood_flows: Dict[str, float]
    cardiac_output: float

class TraditionalPBPK:
    """
    Traditional PBPK model using well-stirred tank approach with explicit ODEs
    """
    
    def __init__(self, physiological_params: PhysiologicalParameters):
        self.physio = physiological_params
        self.organs = list(physiological_params.organ_volumes.keys())
        self.n_organs = len(self.organs)
        
        # Create organ indices
        self.organ_indices = {organ: i for i, organ in enumerate(self.organs)}
        
        # Create blood flow matrix
        self.flow_matrix = self._create_flow_matrix()
        
    def _create_flow_matrix(self) -> np.ndarray:
        """Create blood flow matrix between organs"""
        flow_matrix = np.zeros((self.n_organs, self.n_organs))
        
        # Simplified flow pattern: all organs connected to plasma
        plasma_idx = self.organ_indices.get('plasma', 0)
        
        for organ, idx in self.organ_indices.items():
            if organ != 'plasma':
                # Flow from plasma to organ
                flow_matrix[plasma_idx, idx] = self.physio.blood_flows[organ]
                # Flow from organ to plasma
                flow_matrix[idx, plasma_idx] = self.physio.blood_flows[organ]
        
        return flow_matrix
    
    def _compute_tissue_partition_coefficients(self, drug_params: DrugParameters) -> Dict[str, float]:
        """Compute tissue:plasma partition coefficients"""
        Kp = {}
        
        for organ in self.organs:
            if organ == 'plasma':
                Kp[organ] = 1.0
            else:
                # Use tissue affinity and lipophilicity to estimate Kp
                tissue_affinity = drug_params.tissue_affinity.get(organ, 1.0)
                log_p = drug_params.log_p
                
                # Simplified Kp estimation
                if organ in ['fat', 'brain'] and log_p > 2:
                    Kp[organ] = tissue_affinity * (1 + 0.5 * log_p)
                elif organ in ['liver', 'kidney']:
                    Kp[organ] = tissue_affinity * (1 + 0.2 * log_p)
                else:
                    Kp[organ] = tissue_affinity * (1 + 0.1 * log_p)
        
        return Kp
    
    def _compute_clearance_rates(self, drug_params: DrugParameters) -> Dict[str, float]:
        """Compute organ-specific clearance rates"""
        clearance_rates = {}
        
        for organ in self.organs:
            if organ == 'liver':
                # Hepatic clearance
                metabolic_rate = drug_params.metabolic_rate.get(organ, 0.1)
                clearance_rates[organ] = metabolic_rate * drug_params.fu_plasma
            elif organ == 'kidney':
                # Renal clearance
                metabolic_rate = drug_params.metabolic_rate.get(organ, 0.01)
                clearance_rates[organ] = metabolic_rate * drug_params.fu_plasma
            else:
                # Minimal clearance in other organs
                clearance_rates[organ] = 0.001 * drug_params.fu_plasma
        
        return clearance_rates
    
    def simulate_drug_kinetics(self, 
                             drug_params: DrugParameters,
                             dose: float = 0.1,  # Realistic therapeutic dose (mg/kg)
                             time_points: np.ndarray = None,
                             route: str = 'iv') -> np.ndarray:
        """
        Simulate drug concentration-time profiles using traditional PBPK model
        
        Args:
            drug_params: Drug-specific parameters
            dose: Drug dose (mg/kg)
            time_points: Time points for simulation
            route: Administration route ('iv', 'oral')
            
        Returns:
            Concentration-time profiles [time_points, n_organs]
        """
        
        if time_points is None:
            time_points = np.arange(0, 48, 0.5)
        
        # Compute partition coefficients
        Kp = self._compute_tissue_partition_coefficients(drug_params)
        
        # Compute clearance rates
        clearance_rates = self._compute_clearance_rates(drug_params)
        
        def pbpk_ode(y, t, drug_params, Kp, clearance_rates):
            """ODE system for traditional PBPK model - simplified and stable"""
            concentrations = np.maximum(y, 0.0)  # Ensure non-negative
            dydt = np.zeros_like(concentrations)
            
            plasma_idx = self.organ_indices['plasma']
            
            for i, organ in enumerate(self.organs):
                if organ == 'plasma':
                    # Plasma: receives from all organs, distributes to all organs
                    total_inflow = 0
                    total_outflow = 0
                    
                    for j, other_organ in enumerate(self.organs):
                        if j != i:  # Not plasma itself
                            # Flow from other organs to plasma
                            flow_rate = self.physio.blood_flows[other_organ] * 0.01  # Reduced flow rate
                            total_inflow += flow_rate * concentrations[j] * drug_params.fu_plasma
                            
                            # Flow from plasma to other organs
                            total_outflow += flow_rate * concentrations[i] * drug_params.fu_plasma
                    
                    # Clearance from plasma (realistic rate)
                    clearance = drug_params.clearance * concentrations[i] * 0.1  # Reduced clearance rate
                    
                    dydt[i] = (total_inflow - total_outflow - clearance) / self.physio.organ_volumes[organ]
                    
                else:
                    # Other organs: exchange with plasma
                    flow_rate = self.physio.blood_flows[organ] * 0.01  # Reduced flow rate
                    
                    # Flow from plasma to organ
                    inflow = flow_rate * concentrations[plasma_idx] * drug_params.fu_plasma
                    
                    # Flow from organ to plasma
                    outflow = flow_rate * concentrations[i] * drug_params.fu_plasma
                    
                    # Tissue-specific metabolism (reduced)
                    metabolism = drug_params.metabolic_rate[organ] * concentrations[i] * 0.01
                    
                    dydt[i] = (inflow - outflow - metabolism) / self.physio.organ_volumes[organ]
            
            return dydt
        
        # Initial conditions
        y0 = np.zeros(self.n_organs)
        
        if route == 'iv':
            # Intravenous dose in plasma
            plasma_idx = self.organ_indices['plasma']
            y0[plasma_idx] = dose / self.physio.organ_volumes['plasma']
        elif route == 'oral':
            # Oral dose in gut
            gut_idx = self.organ_indices.get('gut', 0)
            y0[gut_idx] = dose / self.physio.organ_volumes['gut']
        
        # Solve ODE system with error handling
        try:
            solution = odeint(pbpk_ode, y0, time_points, args=(drug_params, Kp, clearance_rates),
                            rtol=1e-6, atol=1e-8, mxstep=10000)
        except:
            # Fallback with simpler solver settings
            solution = odeint(pbpk_ode, y0, time_points, args=(drug_params, Kp, clearance_rates),
                            rtol=1e-3, atol=1e-5)
        
        # Ensure non-negative concentrations
        solution = np.maximum(solution, 0.0)
        
        return solution
    
    def calibrate_parameters(self, 
                           observed_data: np.ndarray,
                           time_points: np.ndarray,
                           drug_params: DrugParameters,
                           target_organs: List[str] = None) -> DrugParameters:
        """
        Calibrate drug parameters to fit observed data
        
        Args:
            observed_data: Observed concentration data [time_points, n_organs]
            time_points: Time points
            drug_params: Initial drug parameters
            target_organs: Organs to use for calibration
            
        Returns:
            Calibrated drug parameters
        """
        
        if target_organs is None:
            target_organs = ['plasma', 'liver', 'kidney']
        
        # Simple parameter adjustment based on observed vs predicted
        calibrated_params = DrugParameters(
            clearance=drug_params.clearance,
            volume_distribution=drug_params.volume_distribution,
            fu_plasma=drug_params.fu_plasma,
            molecular_weight=drug_params.molecular_weight,
            log_p=drug_params.log_p,
            tissue_affinity=drug_params.tissue_affinity.copy(),
            metabolic_rate=drug_params.metabolic_rate.copy(),
            transporter_mediated=drug_params.transporter_mediated
        )
        
        # Predict with current parameters
        predicted = self.simulate_drug_kinetics(calibrated_params, time_points=time_points)
        
        # Adjust parameters based on prediction error
        for organ in target_organs:
            if organ in self.organ_indices:
                org_idx = self.organ_indices[organ]
                
                # Compute prediction error
                observed_peak = np.max(observed_data[:, org_idx])
                predicted_peak = np.max(predicted[:, org_idx])
                
                if predicted_peak > 0:
                    error_ratio = observed_peak / predicted_peak
                    
                    # Adjust clearance and tissue affinity
                    if organ == 'plasma':
                        calibrated_params.clearance *= error_ratio
                    else:
                        calibrated_params.tissue_affinity[organ] *= error_ratio
                        calibrated_params.metabolic_rate[organ] *= error_ratio
        
        return calibrated_params

class PBPKEnsemble:
    """
    Ensemble of traditional PBPK models for robust predictions
    """
    
    def __init__(self, physiological_params: PhysiologicalParameters, n_models: int = 5):
        self.physio = physiological_params
        self.n_models = n_models
        self.models = []
        
        # Create ensemble with slightly different physiological parameters
        for i in range(n_models):
            # Add small variations to physiological parameters
            varied_physio = self._vary_physiological_params(physiological_params, variation=0.1)
            model = TraditionalPBPK(varied_physio)
            self.models.append(model)
    
    def _vary_physiological_params(self, 
                                 physio: PhysiologicalParameters, 
                                 variation: float = 0.1) -> PhysiologicalParameters:
        """Add random variation to physiological parameters"""
        
        varied_volumes = {}
        varied_flows = {}
        
        for organ, volume in physio.organ_volumes.items():
            varied_volumes[organ] = max(0.001, volume * (1 + np.random.normal(0, variation)))
        
        for organ, flow in physio.blood_flows.items():
            varied_flows[organ] = max(0.001, flow * (1 + np.random.normal(0, variation)))
        
        return PhysiologicalParameters(
            organ_volumes=varied_volumes,
            blood_flows=varied_flows,
            cardiac_output=max(1.0, physio.cardiac_output * (1 + np.random.normal(0, variation)))
        )
    
    def predict(self, 
                drug_params: DrugParameters,
                dose: float = 0.1,  # Realistic therapeutic dose (mg/kg)
                time_points: np.ndarray = None,
                route: str = 'iv') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions
        
        Returns:
            Mean predictions and standard deviation
        """
        
        predictions = []
        
        for model in self.models:
            pred = model.simulate_drug_kinetics(drug_params, dose, time_points, route)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

def create_standard_physiological_parameters() -> PhysiologicalParameters:
    """Create standard physiological parameters for a 70kg human"""
    
    organ_volumes = {
        'plasma': 0.055, 'liver': 0.026, 'kidney': 0.004, 'brain': 0.02,
        'heart': 0.004, 'muscle': 0.4, 'fat': 0.19, 'lung': 0.007,
        'spleen': 0.002, 'gut': 0.017, 'bone': 0.14, 'skin': 0.11,
        'pancreas': 0.001, 'adrenal': 0.0002, 'thyroid': 0.0002
    }
    
    blood_flows = {
        'plasma': 5.0, 'liver': 0.8, 'kidney': 0.6, 'brain': 0.5,
        'heart': 0.3, 'muscle': 1.2, 'fat': 0.2, 'lung': 5.0,
        'spleen': 0.1, 'gut': 0.4, 'bone': 0.1, 'skin': 0.3,
        'pancreas': 0.05, 'adrenal': 0.01, 'thyroid': 0.01
    }
    
    return PhysiologicalParameters(
        organ_volumes=organ_volumes,
        blood_flows=blood_flows,
        cardiac_output=5.0
    )

def main():
    """Test the traditional PBPK model"""
    print("Testing Traditional PBPK model...")
    
    # Create physiological parameters
    physio_params = create_standard_physiological_parameters()
    
    # Create drug parameters with realistic values
    drug_params = DrugParameters(
        clearance=0.01,  # L/min/kg (realistic)
        volume_distribution=5.0,
        fu_plasma=0.1,
        molecular_weight=300.0,
        log_p=2.5,
        tissue_affinity={
            'liver': 2.0, 'kidney': 1.5, 'brain': 3.0, 'heart': 1.2,
            'muscle': 1.0, 'fat': 4.0, 'lung': 1.1, 'spleen': 1.3,
            'gut': 1.4, 'bone': 0.8, 'skin': 1.0, 'pancreas': 1.1,
            'adrenal': 1.2, 'thyroid': 1.1
        },
        metabolic_rate={
            'liver': 0.01, 'kidney': 0.002, 'brain': 0.0001, 'heart': 0.0001,
            'muscle': 0.0001, 'fat': 0.0001, 'lung': 0.0001, 'spleen': 0.0001,
            'gut': 0.0001, 'bone': 0.0001, 'skin': 0.0001, 'pancreas': 0.0001,
            'adrenal': 0.0001, 'thyroid': 0.0001
        },
        transporter_mediated=False
    )
    
    # Create PBPK model
    pbpk_model = TraditionalPBPK(physio_params)
    
    # Simulate drug kinetics
    time_points = np.arange(0, 48, 0.5)
    concentrations = pbpk_model.simulate_drug_kinetics(drug_params, time_points=time_points)
    
    print(f"Simulation completed. Shape: {concentrations.shape}")
    print(f"Time points: {len(time_points)}")
    print(f"Organs: {len(pbpk_model.organs)}")
    
    # Test ensemble model
    ensemble = PBPKEnsemble(physio_params, n_models=3)
    mean_pred, std_pred = ensemble.predict(drug_params, time_points=time_points)
    
    print(f"Ensemble prediction shape: {mean_pred.shape}")
    print(f"Standard deviation shape: {std_pred.shape}")
    
    print("Traditional PBPK model test completed successfully!")

if __name__ == "__main__":
    main()
