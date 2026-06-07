"""
Data loading and preprocessing pipeline for WC26 Predictor.

This module provides functions to load, validate, and preprocess training and test data
for the FIFA World Cup 2026 predictor model.

Expected data structure:
- train (1).csv: 1000 rows with 25 features + 'winner' target
- test (2).csv: 250 rows with 25 features (no target)

Features (25 total):
  1. team_name
  2. country_code
  3. confederation
  4-25. 22 numerical features (FIFA rank, player ratings, performance metrics, etc.)
"""

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional


# Define the expected feature columns (all except 'winner')
BASE_FEATURES = [
    'team_name', 'country_code', 'confederation',
    'fifa_rank', 'fifa_points', 'wins_last_10_matches',
    'losses_last_10_matches', 'draws_last_10_matches',
    'win_rate_last_year', 'goals_scored_avg', 'goals_conceded_avg',
    'clean_sheets_last_10', 'shots_per_game', 'shots_on_target_ratio',
    'avg_player_rating', 'star_players_count', 'market_value_million_eur',
    'experience_avg_caps', 'coach_experience_years', 'recent_form_score',
    'possession_avg', 'passing_accuracy', 'host_advantage',
    'travel_distance_avg', 'climate_similarity_score'
]

TARGET_COLUMN = 'winner'


def _get_data_path(filename: str) -> Path:
    """
    Get the full path to a data file.
    
    Handles relative paths from the src/ directory.
    
    Args:
        filename: Name of the CSV file (e.g., 'train (1).csv')
    
    Returns:
        Path object pointing to the data file
    """
    # Get the directory where this script is located
    src_dir = Path(__file__).parent
    data_dir = src_dir.parent / 'data'
    return data_dir / filename


def load_train_data(fillna_method: str = 'mean') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess training data.
    
    Loads train (1).csv, separates features from target, and handles missing values.
    
    Args:
        fillna_method: Method to handle missing values ('mean', 'median', or 'drop')
    
    Returns:
        Tuple of (X_train, y_train):
            - X_train: DataFrame with 25 feature columns, 1000 rows
            - y_train: Series with target variable (binary 0/1), 1000 rows
    
    Raises:
        FileNotFoundError: If train (1).csv is not found
        ValueError: If data shape or columns are invalid
    """
    filepath = _get_data_path('train (1).csv')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Training data not found at {filepath}")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Validate structure
    if len(df) != 1000:
        raise ValueError(f"Expected 1000 training samples, got {len(df)}")
    
    if len(df.columns) != 26:
        raise ValueError(f"Expected 26 columns (25 features + target), got {len(df.columns)}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data")
    
    # Separate features and target
    X_train = df[BASE_FEATURES].copy()
    y_train = df[TARGET_COLUMN].copy()
    
    # Handle missing values
    if fillna_method == 'mean':
        # Fill with mean of each column (only for numeric columns)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
    elif fillna_method == 'median':
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    elif fillna_method == 'drop':
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
    else:
        raise ValueError(f"Unknown fillna_method: {fillna_method}")
    
    # Validate no NaN values remain
    if X_train.isnull().any().any():
        raise ValueError(f"NaN values detected after preprocessing in features")
    if y_train.isnull().any():
        raise ValueError(f"NaN values detected after preprocessing in target")
    
    # Validate shapes
    if X_train.shape != (1000, 25):
        raise ValueError(f"X_train shape mismatch: expected (1000, 25), got {X_train.shape}")
    if y_train.shape != (1000,):
        raise ValueError(f"y_train shape mismatch: expected (1000,), got {y_train.shape}")
    
    return X_train, y_train


def load_test_data(fillna_method: str = 'mean') -> pd.DataFrame:
    """
    Load and preprocess test data.
    
    Loads test (2).csv and handles missing values.
    Note: Test data does not include the target variable.
    
    Args:
        fillna_method: Method to handle missing values ('mean', 'median', or 'drop')
    
    Returns:
        X_test: DataFrame with 25 feature columns, 250 rows
    
    Raises:
        FileNotFoundError: If test (2).csv is not found
        ValueError: If data shape or columns are invalid
    """
    filepath = _get_data_path('test (2).csv')
    
    if not filepath.exists():
        raise FileNotFoundError(f"Test data not found at {filepath}")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Validate structure
    if len(df) != 250:
        raise ValueError(f"Expected 250 test samples, got {len(df)}")
    
    if len(df.columns) != 25:
        raise ValueError(f"Expected 25 columns in test data, got {len(df.columns)}")
    
    # Get features
    X_test = df[BASE_FEATURES].copy()
    
    # Handle missing values
    if fillna_method == 'mean':
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())
    elif fillna_method == 'median':
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].median())
    elif fillna_method == 'drop':
        X_test = X_test.dropna()
    else:
        raise ValueError(f"Unknown fillna_method: {fillna_method}")
    
    # Validate no NaN values remain
    if X_test.isnull().any().any():
        raise ValueError(f"NaN values detected after preprocessing in test features")
    
    # Validate shape
    if X_test.shape != (250, 25):
        raise ValueError(f"X_test shape mismatch: expected (250, 25), got {X_test.shape}")
    
    return X_test


def get_data_quality_report(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: Optional[pd.DataFrame] = None) -> dict:
    """
    Generate a comprehensive data quality report.
    
    Includes shape validation, data types, basic statistics, and missing value checks.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Optional test feature matrix
    
    Returns:
        Dictionary containing data quality metrics and statistics
    """
    report = {
        'train_shape': X_train.shape,
        'train_target_shape': y_train.shape,
        'missing_values_train': X_train.isnull().sum().sum(),
        'missing_values_target': y_train.isnull().sum(),
        'data_types': X_train.dtypes.value_counts().to_dict(),
        'numeric_columns': len(X_train.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(X_train.select_dtypes(include=['object']).columns),
        'train_statistics': {
            'numeric_features': {}
        }
    }
    
    # Collect statistics for numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report['train_statistics']['numeric_features'][col] = {
            'mean': float(X_train[col].mean()),
            'std': float(X_train[col].std()),
            'min': float(X_train[col].min()),
            'max': float(X_train[col].max()),
            'median': float(X_train[col].median())
        }
    
    # Target statistics
    report['target_statistics'] = {
        'class_0_count': int((y_train == 0).sum()),
        'class_1_count': int((y_train == 1).sum()),
        'class_distribution': {
            '0': float((y_train == 0).sum() / len(y_train)),
            '1': float((y_train == 1).sum() / len(y_train))
        }
    }
    
    # Test data statistics if provided
    if X_test is not None:
        report['test_shape'] = X_test.shape
        report['missing_values_test'] = X_test.isnull().sum().sum()
    
    return report


def save_data(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
              output_dir: Optional[str] = None) -> None:
    """
    Save processed data to pickle files for later use.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        output_dir: Directory to save pickle files (defaults to src/../processed_data/)
    
    Raises:
        IOError: If unable to write files
    """
    if output_dir is None:
        src_dir = Path(__file__).parent
        output_dir = src_dir.parent / 'processed_data'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data to pickle files
    with open(output_dir / 'X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    
    with open(output_dir / 'y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(output_dir / 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    
    print(f"Data saved to {output_dir}/")


def load_processed_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load previously saved processed data from pickle files.
    
    Args:
        data_dir: Directory containing pickle files (defaults to src/../processed_data/)
    
    Returns:
        Tuple of (X_train, y_train, X_test)
    
    Raises:
        FileNotFoundError: If pickle files are not found
    """
    if data_dir is None:
        src_dir = Path(__file__).parent
        data_dir = src_dir.parent / 'processed_data'
    
    data_dir = Path(data_dir)
    
    with open(data_dir / 'X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    
    with open(data_dir / 'y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    with open(data_dir / 'X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    
    return X_train, y_train, X_test


def print_data_summary(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> None:
    """
    Print a summary of the loaded data.
    
    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
    """
    print("\n" + "=" * 70)
    print("DATA PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Training Data:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - Missing values in X_train: {X_train.isnull().sum().sum()}")
    print(f"   - Missing values in y_train: {y_train.isnull().sum()}")
    print(f"   - Target class distribution: {(y_train == 0).sum()} (class 0), {(y_train == 1).sum()} (class 1)")
    
    print(f"\n✅ Test Data:")
    print(f"   - X_test shape: {X_test.shape}")
    print(f"   - Missing values in X_test: {X_test.isnull().sum().sum()}")
    
    print(f"\n✅ Features ({len(X_train.columns)} total):")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    print(f"   - Numeric features: {len(numeric_cols)}")
    print(f"   - Categorical features: {len(categorical_cols)}")
    
    print(f"\n✅ Numeric Feature Statistics (X_train):")
    stats_df = X_train[numeric_cols].describe().round(3)
    print(stats_df.to_string())
    
    print("\n" + "=" * 70)
    print("✅ Data pipeline ready: X_train (1000×25), y_train (1000,), X_test (250×25)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # Example usage and testing
    print("Loading data...")
    X_train, y_train = load_train_data()
    X_test = load_test_data()
    
    # Print summary
    print_data_summary(X_train, y_train, X_test)
    
    # Generate and print quality report
    report = get_data_quality_report(X_train, y_train, X_test)
    
    print("\nData Quality Report:")
    print(f"Training samples: {report['train_shape'][0]}")
    print(f"Feature columns: {report['train_shape'][1]}")
    print(f"Test samples: {report['test_shape'][0]}")
    print(f"Target distribution: {report['target_statistics']['class_distribution']}")
