"""
Feature engineering pipeline for WC26 Predictor.

This module implements 8 engineered composite features that combine the 25 base features
to create higher-level indicators of team strength, form, and tactical characteristics.

Engineered Features (8 total):
  1. strength_index: Overall team strength composite
  2. goal_efficiency: Offensive output vs defensive vulnerability
  3. attack_potency: Threat level combining volume, accuracy, and conversion
  4. defensive_solidity: Defensive stability normalized by goals conceded
  5. squad_quality: Weighted combination of player quality, investment, and experience
  6. form_consistency: Interaction between long-term and short-term form
  7. possession_dominance: Ball control effectiveness
  8. star_power: Elite talent concentration
  
Output: 33 features total (25 base + 8 engineered)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Feature engineering pipeline for WC26 Predictor.
    
    Transforms raw features (25) into an expanded feature set (33) by adding
    8 composite engineered features. Handles normalization and edge cases
    (zero/near-zero values, extreme outliers, etc.).
    
    Attributes:
        scaler: StandardScaler instance for optional feature normalization
        fit_stats: Dictionary storing statistics learned during fit
    """
    
    def __init__(self, scale_engineered: bool = False):
        """
        Initialize the feature engineer.
        
        Args:
            scale_engineered: If True, scale engineered features to match base feature scale
        """
        self.scale_engineered = scale_engineered
        self.scaler = StandardScaler() if scale_engineered else None
        self.fit_stats = {}
        self.engineered_columns = [
            'strength_index',
            'goal_efficiency',
            'attack_potency',
            'defensive_solidity',
            'squad_quality',
            'form_consistency',
            'possession_dominance',
            'star_power'
        ]
    
    def fit(self, X_train: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn normalization parameters from training data.
        
        Computes statistics (mean, std) for engineered features to use during transform.
        
        Args:
            X_train: Training feature matrix (25 columns)
        
        Returns:
            self: Returns self for method chaining
        """
        # Engineer features to compute statistics
        X_engineered = engineer_features(X_train)
        
        # Store statistics for engineered features
        engineered = X_engineered[self.engineered_columns]
        self.fit_stats = {
            'means': engineered.mean(),
            'stds': engineered.std(),
            'mins': engineered.min(),
            'maxs': engineered.max()
        }
        
        if self.scale_engineered:
            self.scaler.fit(engineered)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to data.
        
        Adds 8 engineered features to the original 25 base features.
        
        Args:
            X: Feature matrix with 25 columns
        
        Returns:
            DataFrame with 33 columns (25 base + 8 engineered)
        """
        # Engineer features
        X_engineered = engineer_features(X)
        
        # Optionally scale engineered features
        if self.scale_engineered and self.scaler is not None:
            engineered_subset = X_engineered[self.engineered_columns].copy()
            scaled = self.scaler.transform(engineered_subset)
            X_engineered[self.engineered_columns] = scaled
        
        return X_engineered
    
    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X_train: Training feature matrix
        
        Returns:
            Transformed feature matrix with engineering applied
        """
        return self.fit(X_train).transform(X_train)


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add 8 engineered composite features to base features.
    
    Creates higher-level features combining multiple base metrics to capture
    different dimensions of team strength and performance characteristics.
    
    Args:
        X: DataFrame with 25 base feature columns
    
    Returns:
        DataFrame with 33 columns (25 base + 8 engineered)
    
    Raises:
        ValueError: If required columns are missing
    """
    X_out = X.copy()
    
    # Verify required columns exist
    required_cols = {
        'fifa_points', 'recent_form_score', 'avg_player_rating',
        'goals_scored_avg', 'goals_conceded_avg', 'shots_per_game',
        'shots_on_target_ratio', 'clean_sheets_last_10',
        'market_value_million_eur', 'experience_avg_caps',
        'win_rate_last_year', 'possession_avg', 'passing_accuracy',
        'star_players_count', 'host_advantage', 'climate_similarity_score',
        'travel_distance_avg'
    }
    
    missing = required_cols - set(X_out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # 1. STRENGTH_INDEX
    # Composite measure of team's overall strength
    # Formula: fifa_points + recent_form_score*50 + avg_player_rating*10
    X_out['strength_index'] = (
        X_out['fifa_points'] +
        X_out['recent_form_score'] * 50 +
        X_out['avg_player_rating'] * 10
    )
    
    # 2. GOAL_EFFICIENCY
    # Ratio of offensive output to defensive vulnerability
    # Formula: goals_scored_avg / (goals_conceded_avg + 0.01)
    # Adding small constant to avoid division by zero
    X_out['goal_efficiency'] = (
        X_out['goals_scored_avg'] / 
        (X_out['goals_conceded_avg'] + 0.01)
    )
    
    # 3. ATTACK_POTENCY
    # Threat level combining volume, accuracy, and conversion
    # Formula: shots_per_game * shots_on_target_ratio * goals_scored_avg
    X_out['attack_potency'] = (
        X_out['shots_per_game'] *
        X_out['shots_on_target_ratio'] *
        X_out['goals_scored_avg']
    )
    
    # 4. DEFENSIVE_SOLIDITY
    # Measure defensive stability normalized by goals conceded
    # Formula: clean_sheets_last_10 / (goals_conceded_avg + 0.5)
    X_out['defensive_solidity'] = (
        X_out['clean_sheets_last_10'] /
        (X_out['goals_conceded_avg'] + 0.5)
    )
    
    # 5. SQUAD_QUALITY
    # Weighted combination of player quality, investment, and experience
    # Formula: avg_player_rating*0.4 + market_value_million_eur*0.3 + experience_avg_caps*0.3
    X_out['squad_quality'] = (
        X_out['avg_player_rating'] * 0.4 +
        X_out['market_value_million_eur'] * 0.3 +
        X_out['experience_avg_caps'] * 0.3
    )
    
    # 6. FORM_CONSISTENCY
    # Interaction between long-term and short-term form
    # Formula: win_rate_last_year * recent_form_score
    X_out['form_consistency'] = (
        X_out['win_rate_last_year'] *
        X_out['recent_form_score']
    )
    
    # 7. POSSESSION_DOMINANCE
    # Ball control effectiveness (possession × precision)
    # Formula: possession_avg * passing_accuracy
    X_out['possession_dominance'] = (
        X_out['possession_avg'] *
        X_out['passing_accuracy']
    )
    
    # 8. STAR_POWER
    # Elite talent concentration
    # Formula: star_players_count * market_value_million_eur
    X_out['star_power'] = (
        X_out['star_players_count'] *
        X_out['market_value_million_eur']
    )
    
    return X_out


def scale_features(X: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
    """
    Optional: Normalize engineered features to similar scale as base features.
    
    Applies robust or standard scaling to engineered features to prevent
    them from dominating models due to different scales.
    
    Args:
        X: DataFrame with 33 columns (25 base + 8 engineered)
        method: Scaling method ('robust' or 'standard')
    
    Returns:
        DataFrame with scaled engineered features
    
    Raises:
        ValueError: If method is not recognized
    """
    if method not in ['robust', 'standard']:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_out = X.copy()
    engineered_cols = [
        'strength_index', 'goal_efficiency', 'attack_potency',
        'defensive_solidity', 'squad_quality', 'form_consistency',
        'possession_dominance', 'star_power'
    ]
    
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_out[engineered_cols] = scaler.fit_transform(X_out[engineered_cols])
    else:  # robust
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_out[engineered_cols] = scaler.fit_transform(X_out[engineered_cols])
    
    return X_out


def validate_features(X: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Validate engineered features for data quality.
    
    Checks for NaN, inf values, and reports basic statistics.
    
    Args:
        X: DataFrame with 33 columns
        verbose: If True, print validation report
    
    Returns:
        Dictionary with validation results
    
    Raises:
        ValueError: If critical data quality issues are found
    """
    engineered_cols = [
        'strength_index', 'goal_efficiency', 'attack_potency',
        'defensive_solidity', 'squad_quality', 'form_consistency',
        'possession_dominance', 'star_power'
    ]
    
    report = {
        'total_rows': len(X),
        'total_columns': len(X.columns),
        'expected_columns': 33,
        'nan_count': 0,
        'inf_count': 0,
        'feature_stats': {}
    }
    
    # Check for NaN values
    nan_count = X.isnull().sum().sum()
    report['nan_count'] = int(nan_count)
    
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaN values in engineered features")
    
    # Check for inf values
    inf_mask = np.isinf(X.select_dtypes(include=[np.number]))
    inf_count = inf_mask.sum().sum()
    report['inf_count'] = int(inf_count)
    
    if inf_count > 0:
        raise ValueError(f"Found {inf_count} infinite values in engineered features")
    
    # Statistics for engineered features
    for col in engineered_cols:
        if col in X.columns:
            report['feature_stats'][col] = {
                'mean': float(X[col].mean()),
                'std': float(X[col].std()),
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'median': float(X[col].median())
            }
    
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE VALIDATION REPORT")
        print("=" * 70)
        print(f"\n✅ Data Shape: {report['total_rows']} rows × {report['total_columns']} columns")
        print(f"✅ Expected Columns: {report['expected_columns']}")
        print(f"✅ NaN Values: {report['nan_count']}")
        print(f"✅ Inf Values: {report['inf_count']}")
        
        print(f"\n📊 Engineered Feature Statistics:")
        print("-" * 70)
        for col in engineered_cols:
            if col in report['feature_stats']:
                stats = report['feature_stats'][col]
                print(f"\n{col}:")
                print(f"  Mean: {stats['mean']:>12.4f}  |  Std: {stats['std']:>12.4f}")
                print(f"  Min:  {stats['min']:>12.4f}  |  Max: {stats['max']:>12.4f}")
                print(f"  Median: {stats['median']:>12.4f}")
        print("\n" + "=" * 70)
    
    return report


def compute_feature_correlations(X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute correlation matrix for engineered features.
    
    Identifies which engineered features are correlated with each other,
    helping ensure diversity in the feature set.
    
    Args:
        X: DataFrame with engineered features
        verbose: If True, print correlation analysis
    
    Returns:
        Correlation matrix for engineered features
    """
    engineered_cols = [
        'strength_index', 'goal_efficiency', 'attack_potency',
        'defensive_solidity', 'squad_quality', 'form_consistency',
        'possession_dominance', 'star_power'
    ]
    
    # Compute correlation matrix
    corr_matrix = X[engineered_cols].corr()
    
    if verbose:
        print("\n" + "=" * 70)
        print("ENGINEERED FEATURES CORRELATION MATRIX")
        print("=" * 70)
        print(corr_matrix.round(3).to_string())
        
        print("\n📊 Top Correlated Feature Pairs:")
        print("-" * 70)
        
        # Get upper triangle of correlation matrix (to avoid duplicates)
        corr_pairs = []
        for i in range(len(engineered_cols)):
            for j in range(i+1, len(engineered_cols)):
                corr_pairs.append({
                    'feature1': engineered_cols[i],
                    'feature2': engineered_cols[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation value
        corr_pairs_sorted = sorted(
            corr_pairs,
            key=lambda x: abs(x['correlation']),
            reverse=True
        )
        
        # Print top 10
        for pair in corr_pairs_sorted[:10]:
            print(f"{pair['feature1']:25s} <-> {pair['feature2']:25s}: {pair['correlation']:>8.4f}")
        
        print("\n" + "=" * 70)
    
    return corr_matrix


if __name__ == '__main__':
    """
    Test feature engineering pipeline.
    """
    print("Testing feature engineering pipeline...\n")
    
    # Import data loader
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_train_data, load_test_data
    
    # Load data
    print("📥 Loading training and test data...")
    X_train, y_train = load_train_data()
    X_test = load_test_data()
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ X_test: {X_test.shape}")
    
    # Test FeatureEngineer class
    print("\n🔧 Initializing FeatureEngineer...")
    fe = FeatureEngineer(scale_engineered=False)
    
    print("🔧 Fitting on training data...")
    fe.fit(X_train)
    
    print("🔧 Transforming training data...")
    X_train_engineered = fe.transform(X_train)
    
    print("🔧 Transforming test data...")
    X_test_engineered = fe.transform(X_test)
    
    # Validate outputs
    print("\n✅ Feature Engineering Complete!")
    print(f"   X_train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"   X_train_engineered: {X_train_engineered.shape[0]} samples × {X_train_engineered.shape[1]} features")
    print(f"   X_test_engineered: {X_test_engineered.shape[0]} samples × {X_test_engineered.shape[1]} features")
    
    # Validate data quality
    print("\n🔍 Validating feature quality...")
    report = validate_features(X_train_engineered, verbose=True)
    
    # Compute correlations
    print("\n🔗 Computing feature correlations...")
    corr_matrix = compute_feature_correlations(X_train_engineered, verbose=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING PIPELINE TEST SUCCESSFUL")
    print("=" * 70)
    print(f"\n✅ Features engineered: 25 base → 33 total, no NaN values")
    print(f"✅ Training set: {X_train_engineered.shape}")
    print(f"✅ Test set: {X_test_engineered.shape}")
    print(f"✅ All quality checks passed")
    print("\n" + "=" * 70 + "\n")
