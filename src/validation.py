"""
K-fold Cross-Validation Strategy for WC26 Predictor
Implements comprehensive stratified cross-validation with detailed metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, log_loss, confusion_matrix, roc_curve, auc,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


class CVValidator:
    """Comprehensive K-fold cross-validation validator for ensemble models."""
    
    def __init__(self, model, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize CVValidator.
        
        Parameters:
        -----------
        model : object
            Model with fit(), predict(), predict_proba() methods
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
        """
        self.model = model
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Storage for CV results
        self.cv_results = {}
        self.fold_metrics = []
        self.fold_predictions = []
        self.fold_probabilities = []
        self.fold_true_labels = []
        self.fold_indices = []
        
    def run_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Run stratified K-fold cross-validation with comprehensive metrics.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features (n_samples, n_features)
        y : pd.Series
            Training labels (n_samples,)
        cv : int
            Number of folds (default: 5)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with CV results, fold metrics, and statistics
        """
        print(f"\n{'='*70}")
        print(f"Starting {cv}-Fold Stratified Cross-Validation")
        print(f"Total samples: {len(X)}, Class distribution: {y.value_counts().to_dict()}")
        print(f"{'='*70}\n")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Initialize metrics storage
        metrics_list = {
            'accuracy': [], 'auc_roc': [], 'precision': [], 
            'recall': [], 'f1': [], 'log_loss': []
        }
        
        fold_counter = 0
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_counter = fold + 1
            
            # Split data
            X_train_fold = X.iloc[train_idx].reset_index(drop=True)
            X_test_fold = X.iloc[test_idx].reset_index(drop=True)
            y_train_fold = y.iloc[train_idx].reset_index(drop=True)
            y_test_fold = y.iloc[test_idx].reset_index(drop=True)
            
            print(f"Fold {fold_counter}/{cv}:")
            print(f"  Train: {len(X_train_fold)} samples, Class distribution: {y_train_fold.value_counts().to_dict()}")
            print(f"  Test:  {len(X_test_fold)} samples, Class distribution: {y_test_fold.value_counts().to_dict()}")
            
            # Train model on this fold
            fold_model = clone(self.model)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            y_pred = fold_model.predict(X_test_fold)
            y_proba = fold_model.predict_proba(X_test_fold)
            
            # Ensure y_proba is 1D (for binary classification)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test_fold, y_pred),
                'auc_roc': roc_auc_score(y_test_fold, y_proba),
                'precision': precision_score(y_test_fold, y_pred, zero_division=0),
                'recall': recall_score(y_test_fold, y_pred, zero_division=0),
                'f1': f1_score(y_test_fold, y_pred, zero_division=0),
                'log_loss': log_loss(y_test_fold, y_proba),
            }
            
            # Store results
            self.fold_metrics.append(metrics)
            self.fold_predictions.append(y_pred)
            self.fold_probabilities.append(y_proba)
            self.fold_true_labels.append(y_test_fold.values)
            self.fold_indices.append(test_idx)
            
            # Append to metrics list
            for key, value in metrics.items():
                metrics_list[key].append(value)
            
            # Print fold metrics
            print(f"  Metrics:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    AUC-ROC:   {metrics['auc_roc']:.4f} ⭐")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
            print(f"    Log Loss:  {metrics['log_loss']:.4f}\n")
        
        # Compute statistics
        self.cv_results = self._compute_statistics(metrics_list)
        
        return self.cv_results
    
    def _compute_statistics(self, metrics_list: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compute mean and std for each metric."""
        cv_results = {}
        
        for metric_name, values in metrics_list.items():
            cv_results[f'{metric_name}_mean'] = np.mean(values)
            cv_results[f'{metric_name}_std'] = np.std(values)
            cv_results[f'{metric_name}_values'] = values
        
        return cv_results
    
    def get_cv_report(self) -> str:
        """
        Generate formatted CV results report.
        
        Returns:
        --------
        str
            Formatted report with statistics and fold details
        """
        if not self.cv_results:
            raise ValueError("Run cross-validation first with run_cross_validation()")
        
        report = []
        report.append("\n" + "="*70)
        report.append("CROSS-VALIDATION RESULTS (5-Fold Stratified)")
        report.append("="*70)
        
        # Main metrics summary
        report.append("\n📊 OVERALL METRICS (Mean ± Std)")
        report.append("-"*70)
        
        metrics_order = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1', 'log_loss']
        
        for metric in metrics_order:
            mean = self.cv_results.get(f'{metric}_mean', 0)
            std = self.cv_results.get(f'{metric}_std', 0)
            
            if metric == 'auc_roc':
                report.append(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f} ⭐ PRIMARY METRIC")
            else:
                report.append(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
        
        # Per-class performance
        report.append("\n📋 PER-CLASS PERFORMANCE")
        report.append("-"*70)
        
        # Aggregate predictions for overall per-class metrics
        y_true_all = np.concatenate(self.fold_true_labels)
        y_pred_all = np.concatenate(self.fold_predictions)
        
        class_report = classification_report(y_true_all, y_pred_all, output_dict=True)
        
        for class_label in [0, 1]:
            precision = class_report[str(class_label)]['precision']
            recall = class_report[str(class_label)]['recall']
            f1 = class_report[str(class_label)]['f1-score']
            support = int(class_report[str(class_label)]['support'])
            
            class_name = "Class 0 (Loss)" if class_label == 0 else "Class 1 (Win)"
            report.append(f"\n{class_name}:")
            report.append(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Support: {support}")
        
        # Fold details
        report.append("\n\n📈 FOLD-WISE RESULTS")
        report.append("-"*70)
        
        for fold_idx, metrics in enumerate(self.fold_metrics, 1):
            report.append(f"Fold {fold_idx}: AUC={metrics['auc_roc']:.4f}, Acc={metrics['accuracy']:.4f}, "
                         f"F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        # Identify best and worst folds
        report.append("\n\n🎯 FOLD RANKING (by AUC-ROC)")
        report.append("-"*70)
        
        fold_aucs = [(i+1, m['auc_roc']) for i, m in enumerate(self.fold_metrics)]
        fold_aucs_sorted = sorted(fold_aucs, key=lambda x: x[1], reverse=True)
        
        for rank, (fold_num, auc_score) in enumerate(fold_aucs_sorted, 1):
            status = "✅ BEST" if rank == 1 else ("⚠️  WORST" if rank == len(fold_aucs_sorted) else "")
            report.append(f"  {rank}. Fold {fold_num}: AUC = {auc_score:.4f} {status}")
        
        # Model stability assessment
        report.append("\n\n🔍 MODEL STABILITY ASSESSMENT")
        report.append("-"*70)
        
        auc_std = self.cv_results.get('auc_roc_std', 0)
        acc_std = self.cv_results.get('accuracy_std', 0)
        
        if auc_std < 0.03:
            stability = "✅ EXCELLENT - Very consistent across folds"
        elif auc_std < 0.05:
            stability = "✅ GOOD - Consistent performance across folds"
        elif auc_std < 0.10:
            stability = "⚠️  FAIR - Some variance but acceptable"
        else:
            stability = "❌ POOR - High variance, possible overfitting"
        
        report.append(f"AUC-ROC Std Dev:   {auc_std:.4f}")
        report.append(f"Accuracy Std Dev:  {acc_std:.4f}")
        report.append(f"Assessment:        {stability}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    def plot_cv_results(self, output_dir: str = "outputs/cv_plots") -> None:
        """
        Generate comprehensive visualization plots.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Metric comparison bar chart
        self._plot_metric_comparison(output_dir)
        
        # 2. AUC-ROC across folds with error bars
        self._plot_fold_auc(output_dir)
        
        # 3. Confusion matrices for each fold
        self._plot_confusion_matrices(output_dir)
        
        # 4. Calibration curves
        self._plot_calibration_curves(output_dir)
        
        # 5. ROC curves for each fold
        self._plot_roc_curves(output_dir)
        
        print(f"\n✅ All plots saved to {output_dir}/")
    
    def _plot_metric_comparison(self, output_dir: str) -> None:
        """Plot comparison of all metrics."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['accuracy', 'auc_roc', 'precision', 'recall', 'f1', 'log_loss']
        means = [self.cv_results.get(f'{m}_mean', 0) for m in metrics]
        stds = [self.cv_results.get(f'{m}_std', 0) for m in metrics]
        
        # Normalize log_loss to 0-1 scale for visualization
        means = [means[i] / 0.7 if i == 5 else means[i] for i in range(len(means))]
        stds = [stds[i] / 0.7 if i == 5 else stds[i] for i in range(len(stds))]
        
        x_pos = np.arange(len(metrics))
        colors = ['#1f77b4' if i != 1 else '#ff7f0e' for i in range(len(metrics))]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Validation Metrics Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics], rotation=45)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, std) in enumerate(zip(bars, stds)):
            actual_mean = self.cv_results.get(f'{metrics[i]}_mean', 0)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{actual_mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cv_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fold_auc(self, output_dir: str) -> None:
        """Plot AUC-ROC for each fold with error bars."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fold_nums = list(range(1, len(self.fold_metrics) + 1))
        auc_scores = [m['auc_roc'] for m in self.fold_metrics]
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        
        # Plot line with markers
        ax.plot(fold_nums, auc_scores, marker='o', markersize=10, linewidth=2, 
               color='#ff7f0e', label='Fold AUC')
        
        # Add mean line
        ax.axhline(y=mean_auc, color='green', linestyle='--', linewidth=2, 
                  label=f'Mean AUC = {mean_auc:.4f}')
        
        # Add error band
        ax.fill_between(fold_nums, mean_auc - std_auc, mean_auc + std_auc, 
                       alpha=0.2, color='green', label=f'±1 Std Dev ({std_auc:.4f})')
        
        ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax.set_title('AUC-ROC Score Across Folds', fontsize=14, fontweight='bold')
        ax.set_xticks(fold_nums)
        ax.set_ylim([min(auc_scores) - 0.05, max(auc_scores) + 0.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add value labels
        for fold, auc in zip(fold_nums, auc_scores):
            ax.text(fold, auc + 0.01, f'{auc:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cv_fold_auc.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, output_dir: str) -> None:
        """Plot confusion matrices for each fold."""
        n_folds = len(self.fold_metrics)
        
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        if n_folds == 1:
            axes = [axes]
        
        for fold_idx, (y_true, y_pred, ax) in enumerate(zip(
            self.fold_true_labels, self.fold_predictions, axes)):
            
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar=False, square=True, annot_kws={'size': 12})
            
            ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
            ax.set_ylabel('True', fontsize=11, fontweight='bold')
            ax.set_title(f'Fold {fold_idx+1} Confusion Matrix\n(AUC: {self.fold_metrics[fold_idx]["auc_roc"]:.4f})',
                        fontsize=11, fontweight='bold')
            ax.set_xticklabels(['Loss', 'Win'])
            ax.set_yticklabels(['Loss', 'Win'])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curves(self, output_dir: str) -> None:
        """Plot calibration curves for each fold."""
        n_folds = len(self.fold_metrics)
        
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        if n_folds == 1:
            axes = [axes]
        
        for fold_idx, (y_true, y_proba, ax) in enumerate(zip(
            self.fold_true_labels, self.fold_probabilities, axes)):
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            
            # Plot calibration curve
            ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Fold Calibration')
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
            ax.set_title(f'Fold {fold_idx+1} Calibration Curve', fontsize=11, fontweight='bold')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/calibration_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, output_dir: str) -> None:
        """Plot ROC curves for each fold."""
        n_folds = len(self.fold_metrics)
        
        fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
        if n_folds == 1:
            axes = [axes]
        
        for fold_idx, (y_true, y_proba, ax) in enumerate(zip(
            self.fold_true_labels, self.fold_probabilities, axes)):
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color='#ff7f0e', linewidth=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.4f})')
            
            # Plot random classifier line
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            ax.set_title(f'Fold {fold_idx+1} ROC Curve', fontsize=11, fontweight='bold')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, output_path: str = "outputs/CV_REPORT.md") -> None:
        """
        Save CV report to markdown file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = self.get_cv_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ CV Report saved to {output_path}")


def run_validation_pipeline(model, X_train, y_train, cv: int = 5, output_dir: str = "outputs") -> CVValidator:
    """
    Run complete validation pipeline and generate all outputs.
    
    Parameters:
    -----------
    model : object
        Model with fit(), predict(), predict_proba() methods
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of folds
    output_dir : str
        Output directory for reports and plots
        
    Returns:
    --------
    CVValidator
        Validator object with results
    """
    # Initialize validator
    validator = CVValidator(model, random_state=42)
    
    # Run cross-validation
    cv_results = validator.run_cross_validation(X_train, y_train, cv=cv)
    
    # Generate report
    print(validator.get_cv_report())
    
    # Save report
    validator.save_report(f"{output_dir}/CV_REPORT.md")
    
    # Generate plots
    validator.plot_cv_results(f"{output_dir}/cv_plots")
    
    # Print completion message
    mean_auc = cv_results.get('auc_roc_mean', 0)
    std_auc = cv_results.get('auc_roc_std', 0)
    print(f"\n✅ {cv}-Fold CV Complete: Mean AUC-ROC = {mean_auc:.4f} ± {std_auc:.4f}")
    
    return validator
