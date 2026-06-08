
# WC26 Predictor - Feature Engineering Report

## Summary
- **Input Features**: 25 base features
- **Output Features**: 33 total (25 base + 8 engineered)
- **Training Samples**: 1000
- **Test Samples**: 250

## Engineered Features

### 1. strength_index
- **Formula**: `fifa_points + recent_form_score*50 + avg_player_rating*10`
- **Purpose**: Composite measure of team's overall strength
- **Mean**: 2736.69
- **Std**: 280.48
- **Range**: [2193.00, 3364.50]

### 2. goal_efficiency
- **Formula**: `goals_scored_avg / (goals_conceded_avg + 0.01)`
- **Purpose**: Ratio of offensive output to defensive vulnerability
- **Mean**: 1.56
- **Std**: 1.13
- **Range**: [0.23, 6.36]

### 3. attack_potency
- **Formula**: `shots_per_game * shots_on_target_ratio * goals_scored_avg`
- **Purpose**: Threat level combining volume, accuracy, and conversion
- **Mean**: 7.98
- **Std**: 6.13
- **Range**: [0.92, 28.72]

### 4. defensive_solidity
- **Formula**: `clean_sheets_last_10 / (goals_conceded_avg + 0.5)`
- **Purpose**: Measure defensive stability normalized by goals conceded
- **Mean**: 1.85
- **Std**: 1.53
- **Range**: [0.00, 7.61]

### 5. squad_quality
- **Formula**: `avg_player_rating*0.4 + market_value_million_eur*0.3 + experience_avg_caps*0.3`
- **Purpose**: Weighted combination of player quality, investment, and experience
- **Mean**: 197.20
- **Std**: 82.04
- **Range**: [95.46, 413.30]

### 6. form_consistency
- **Formula**: `win_rate_last_year * recent_form_score`
- **Purpose**: Interaction between long-term and short-term form
- **Mean**: 3.28
- **Std**: 1.92
- **Range**: [0.00, 8.40]

### 7. possession_dominance
- **Formula**: `possession_avg * passing_accuracy`
- **Purpose**: Ball control effectiveness (possession × precision)
- **Mean**: 4244.84
- **Std**: 816.90
- **Range**: [2855.25, 6436.92]

### 8. star_power
- **Formula**: `star_players_count * market_value_million_eur`
- **Purpose**: Elite talent concentration
- **Mean**: 1387.83
- **Std**: 1374.43
- **Range**: [0.00, 5975.00]

## Feature Correlations

The engineered features show moderate to strong correlations with each other:

| Feature Pair | Correlation |
|---|---|
| strength_index ↔ form_consistency | 0.9532 |
| squad_quality ↔ star_power | 0.9188 |
| strength_index ↔ possession_dominance | 0.8919 |
| strength_index ↔ attack_potency | 0.8912 |
| goal_efficiency ↔ attack_potency | 0.8874 |

## Quality Assurance

✅ **Data Validation**:
- No NaN values detected
- No infinite values detected
- Output shapes verified: (1000, 33) for train, (250, 33) for test
- Base features preserved and unchanged

✅ **Edge Case Handling**:
- Division by zero protection (goals_conceded_avg + 0.01, + 0.5)
- Zero value handling in all composite features
- Numeric stability verified

✅ **Implementation**:
- FeatureEngineer class with fit/transform/fit_transform methods
- Separate helper functions for validation and correlation analysis
- Comprehensive docstrings and inline comments

## Files Generated

1. **src/features.py** - Main feature engineering module
2. **processed_data/X_train_engineered.pkl** - Engineered training features
3. **processed_data/X_test_engineered.pkl** - Engineered test features
