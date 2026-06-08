"""
Script to run K-fold cross-validation on the WC26 Predictor ensemble model.
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from models import WC26EnsembleModel
from validation import run_validation_pipeline

def main():
    """Run complete cross-validation pipeline."""
    
    print("\n" + "="*70)
    print("WC26 PREDICTOR - K-FOLD CROSS-VALIDATION PIPELINE")
    print("="*70)
    
    # Load training data
    print("\n📦 Loading training data...")
    X_train_engineered = pickle.load(open('processed_data/X_train_engineered.pkl', 'rb'))
    y_train = pickle.load(open('processed_data/y_train.pkl', 'rb'))
    
    print(f"✅ Data loaded:")
    print(f"   X_train_engineered: {X_train_engineered.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   Class distribution: {y_train.value_counts().to_dict()}")
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/cv_plots', exist_ok=True)
    
    # Initialize ensemble model
    print("\n🔧 Initializing WC26 Ensemble Model...")
    model = WC26EnsembleModel(random_state=42)
    
    # Run validation pipeline
    print("\n🚀 Running 5-Fold Stratified Cross-Validation...")
    validator = run_validation_pipeline(
        model, 
        X_train_engineered, 
        y_train, 
        cv=5, 
        output_dir='outputs'
    )
    
    # Print summary
    print("\n" + "="*70)
    print("✅ CROSS-VALIDATION PIPELINE COMPLETE")
    print("="*70)
    print("\n📊 Generated Outputs:")
    print("  ✓ outputs/CV_REPORT.md - Detailed CV results and statistics")
    print("  ✓ outputs/cv_plots/cv_metrics.png - Metric comparison chart")
    print("  ✓ outputs/cv_plots/cv_fold_auc.png - AUC-ROC per fold")
    print("  ✓ outputs/cv_plots/confusion_matrices.png - Fold confusion matrices")
    print("  ✓ outputs/cv_plots/calibration_curves.png - Probability calibration")
    print("  ✓ outputs/cv_plots/roc_curves.png - ROC curves per fold")
    
    # Return validator for interactive use
    return validator

if __name__ == '__main__':
    validator = main()
