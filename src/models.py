"""
WC26 Ensemble Model - XGBoost + Random Forest with Isotonic Calibration
Combines two powerful classifiers for robust probability predictions.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class WC26EnsembleModel:
    """Ensemble model combining XGBoost and Random Forest with calibration."""
    
    def __init__(self, random_state=42):
        """Initialize XGBoost and Random Forest models with optimal parameters."""
        self.random_state = random_state
        
        # XGBoost Classifier
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Random Forest Classifier
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Isotonic calibrator for blended predictions
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
        # Placeholders for training info
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.training_metrics = {}
        self.feature_names = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, val_split: float = 0.2) -> None:
        """
        Train both models, create ensemble, and apply isotonic calibration.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features (1000, 33)
        y_train : pd.Series
            Training labels (1000,)
        val_split : float
            Validation split ratio (default: 0.2 for 80/20 split)
        """
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Stratified train/val split (80/20)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_split, 
            random_state=self.random_state,
            stratify=y_train
        )
        
        self.X_train = X_tr
        self.y_train = y_tr
        self.X_val = X_val
        self.y_val = y_val
        
        print(f"Training XGBoost with {len(X_tr)} samples...")
        
        # Train XGBoost with early stopping on validation set
        eval_set = [(X_val, y_val)]
        self.xgb_model.fit(
            X_tr, y_tr,
            eval_set=eval_set,
            verbose=False
        )
        
        print(f"Training Random Forest with {len(X_tr)} samples...")
        
        # Train Random Forest
        self.rf_model.fit(X_tr, y_tr)
        
        print("Applying isotonic calibration...")
        
        # Get blended predictions on validation set
        xgb_proba_val = self.xgb_model.predict_proba(X_val)[:, 1]
        rf_proba_val = self.rf_model.predict_proba(X_val)[:, 1]
        ensemble_proba_val = 0.6 * xgb_proba_val + 0.4 * rf_proba_val
        
        # Train isotonic calibrator on validation set
        self.calibrator.fit(ensemble_proba_val, y_val)
        
        print("✅ Ensemble model trained: XGB(500) + RF(300), 60/40 blend, isotonic calibrated")
        
        # Compute training metrics
        self._compute_training_metrics()
        
    def _compute_training_metrics(self) -> None:
        """Compute and store training metrics on validation set."""
        y_val_pred = self.predict(self.X_val)
        y_val_proba = self.predict_proba(self.X_val)
        
        self.training_metrics = {
            'val_accuracy': accuracy_score(self.y_val, y_val_pred),
            'val_auc': roc_auc_score(self.y_val, y_val_proba),
            'val_precision': precision_score(self.y_val, y_val_pred),
            'val_recall': recall_score(self.y_val, y_val_pred),
            'val_f1': f1_score(self.y_val, y_val_pred),
        }
        
        # Train metrics
        y_train_pred = self.predict(self.X_train)
        y_train_proba = self.predict_proba(self.X_train)
        
        self.training_metrics.update({
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'train_auc': roc_auc_score(self.y_train, y_train_proba),
        })
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary predictions [0, 1].
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
            
        Returns:
        --------
        np.ndarray
            Binary predictions (shape: (n_samples,))
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated probability scores [0.0-1.0].
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
            
        Returns:
        --------
        np.ndarray
            Calibrated probabilities for class 1 (shape: (n_samples,))
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get blended predictions
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        ensemble_proba = 0.6 * xgb_proba + 0.4 * rf_proba
        
        # Apply calibration
        calibrated_proba = self.calibrator.predict(ensemble_proba)
        
        return calibrated_proba
    
    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Parameters:
        -----------
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation labels
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with accuracy, AUC-ROC, precision, recall, F1
        """
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc_roc': roc_auc_score(y_val, y_proba),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
        }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform stratified 5-fold cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Full training features
        y : pd.Series
            Full training labels
        cv : int
            Number of folds
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with mean and std of AUC-ROC scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        xgb_scores = []
        rf_scores = []
        ensemble_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold models
            xgb_fold = XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, min_child_weight=3, gamma=0.1,
                random_state=self.random_state, eval_metric='logloss', verbosity=0
            )
            xgb_fold.fit(X_fold_train, y_fold_train)
            
            rf_fold = RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            )
            rf_fold.fit(X_fold_train, y_fold_train)
            
            # Blend predictions
            xgb_proba = xgb_fold.predict_proba(X_fold_val)[:, 1]
            rf_proba = rf_fold.predict_proba(X_fold_val)[:, 1]
            ensemble_proba = 0.6 * xgb_proba + 0.4 * rf_proba
            
            # Compute AUC scores
            xgb_auc = roc_auc_score(y_fold_val, xgb_proba)
            rf_auc = roc_auc_score(y_fold_val, rf_proba)
            ensemble_auc = roc_auc_score(y_fold_val, ensemble_proba)
            
            xgb_scores.append(xgb_auc)
            rf_scores.append(rf_auc)
            ensemble_scores.append(ensemble_auc)
            
            print(f"Fold {fold+1}: XGB={xgb_auc:.4f}, RF={rf_auc:.4f}, Ensemble={ensemble_auc:.4f}")
        
        return {
            'xgb_auc_mean': np.mean(xgb_scores),
            'xgb_auc_std': np.std(xgb_scores),
            'rf_auc_mean': np.mean(rf_scores),
            'rf_auc_std': np.std(rf_scores),
            'ensemble_auc_mean': np.mean(ensemble_scores),
            'ensemble_auc_std': np.std(ensemble_scores),
        }
    
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top N important features from XGBoost model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature names and importance scores
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        importance_dict = self.xgb_model.get_booster().get_score(
            importance_type='weight'
        )
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature_name', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df[['feature_name', 'importance']].head(top_n)
    
    def get_model_report(self) -> str:
        """
        Generate comprehensive training report.
        
        Returns:
        --------
        str
            Formatted report with metrics and feature importance
        """
        report = []
        report.append("=" * 70)
        report.append("WC26 ENSEMBLE MODEL - TRAINING REPORT")
        report.append("=" * 70)
        
        report.append("\n📊 MODEL ARCHITECTURE")
        report.append("-" * 70)
        report.append("XGBoost: 500 estimators, LR=0.05, max_depth=6")
        report.append("Random Forest: 300 estimators, max_depth=10")
        report.append("Ensemble: 60% XGB + 40% RF, isotonic calibration")
        
        report.append("\n📈 TRAINING METRICS")
        report.append("-" * 70)
        report.append(f"Train Accuracy:     {self.training_metrics.get('train_accuracy', 0):.4f}")
        report.append(f"Train AUC-ROC:      {self.training_metrics.get('train_auc', 0):.4f}")
        report.append(f"Val Accuracy:       {self.training_metrics.get('val_accuracy', 0):.4f}")
        report.append(f"Val AUC-ROC:        {self.training_metrics.get('val_auc', 0):.4f}")
        report.append(f"Val Precision:      {self.training_metrics.get('val_precision', 0):.4f}")
        report.append(f"Val Recall:         {self.training_metrics.get('val_recall', 0):.4f}")
        report.append(f"Val F1-Score:       {self.training_metrics.get('val_f1', 0):.4f}")
        
        report.append("\n🎯 FEATURE IMPORTANCE (Top 15)")
        report.append("-" * 70)
        feature_imp = self.feature_importance(top_n=15)
        for idx, row in feature_imp.iterrows():
            report.append(f"{row['feature_name']:30s} {row['importance']:8.0f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def save_models(self, models_dir: str = "models") -> None:
        """
        Save trained models with pickle.
        
        Parameters:
        -----------
        models_dir : str
            Directory to save models (default: "models")
        """
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f"{models_dir}/ensemble_model.pkl", 'wb') as f:
            pickle.dump({'xgb': self.xgb_model, 'rf': self.rf_model, 'calibrator': self.calibrator}, f)
        
        with open(f"{models_dir}/xgb_model.pkl", 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        with open(f"{models_dir}/rf_model.pkl", 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        print(f"✅ Models saved to {models_dir}/")
    
    def load_models(self, models_dir: str = "models") -> None:
        """
        Load trained models from pickle files.
        
        Parameters:
        -----------
        models_dir : str
            Directory to load models from (default: "models")
        """
        with open(f"{models_dir}/ensemble_model.pkl", 'rb') as f:
            ensemble_dict = pickle.load(f)
            self.xgb_model = ensemble_dict['xgb']
            self.rf_model = ensemble_dict['rf']
            self.calibrator = ensemble_dict['calibrator']
        
        print(f"✅ Models loaded from {models_dir}/")
