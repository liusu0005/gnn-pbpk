# Dynamic GNN-PBPK Model: Experimental Design

## Overview
This document outlines the experimental framework for validating the Dynamic GNN-PBPK model against traditional PBPK approaches.

## Experimental Objectives
1. **Primary**: Demonstrate that GNN-PBPK models can achieve superior prediction accuracy compared to traditional PBPK models
2. **Secondary**: Show that GNN-PBPK models can capture complex, non-linear pharmacokinetic phenomena without explicit ODE formulation
3. **Tertiary**: Validate the model's ability to generalize across different drug classes and physiological conditions

## Experimental Design

### 1. Dataset Generation
- **Synthetic Preclinical Data**: Generate realistic drug concentration-time profiles for multiple tissues
- **Drug Classes**: Include drugs with different ADME properties (high/low clearance, extensive/poor metabolism, transporter-mediated uptake)
- **Physiological Variability**: Incorporate inter-individual variability in organ volumes, blood flows, and metabolic rates
- **Temporal Resolution**: High-resolution time series (0.1-1 hour intervals) over 24-48 hours

### 2. Physiological Graph Structure
- **Nodes**: 15 major organs/tissues (brain, heart, liver, kidney, muscle, fat, lung, spleen, gut, bone, skin, etc.)
- **Edges**: Blood flow connections based on physiological circulation patterns
- **Node Features**: Organ volume, blood flow rate, metabolic capacity, transporter expression
- **Edge Features**: Blood flow rate, drug binding characteristics

### 3. Model Architecture
- **GNN Type**: Graph Attention Network (GAT) with temporal modeling
- **Message Passing**: Drug concentration information flow between connected organs
- **Node Updates**: Learned functions for drug uptake, metabolism, and efflux
- **Temporal Modeling**: LSTM/GRU layers for time series prediction

### 4. Baseline Comparison
- **Traditional PBPK**: Well-stirred tank model with explicit ODEs
- **Calibration**: Manual parameter tuning for optimal performance
- **Validation**: Same test dataset for fair comparison

### 5. Evaluation Metrics
- **Primary**: Mean Absolute Percentage Error (MAPE) for plasma and tissue concentrations
- **Secondary**: Root Mean Square Error (RMSE), R² correlation coefficient
- **Tertiary**: Prediction accuracy for peak concentrations and elimination half-lives

## Experimental Protocol

### Phase 1: Data Generation and Preprocessing
1. Generate synthetic preclinical datasets for 50 different drugs
2. Create physiological graphs with realistic connectivity
3. Split data: 70% training, 15% validation, 15% testing

### Phase 2: Model Training
1. Train GNN-PBPK model on synthetic data
2. Implement traditional PBPK baseline
3. Hyperparameter optimization for both models

### Phase 3: Validation and Comparison
1. Evaluate both models on test dataset
2. Statistical significance testing
3. Analysis of prediction errors across drug classes

### Phase 4: Ablation Studies
1. Test different GNN architectures
2. Evaluate impact of graph structure variations
3. Assess sensitivity to training data size

## Expected Outcomes
- GNN-PBPK model should achieve >20% reduction in prediction error
- Superior performance on drugs with complex, non-linear kinetics
- Demonstrated ability to capture transporter-mediated effects
- Generalization across different physiological conditions
