======================================================================
WC26 ENSEMBLE MODEL - TRAINING REPORT
======================================================================

📊 MODEL ARCHITECTURE
----------------------------------------------------------------------
XGBoost: 500 estimators, LR=0.05, max_depth=6
Random Forest: 300 estimators, max_depth=10
Ensemble: 60% XGB + 40% RF, isotonic calibration

📈 TRAINING METRICS
----------------------------------------------------------------------
Train Accuracy:     0.9925
Train AUC-ROC:      0.9997
Val Accuracy:       0.6400
Val AUC-ROC:        0.6859
Val Precision:      0.6667
Val Recall:         0.4842
Val F1-Score:       0.5610

🎯 FEATURE IMPORTANCE (Top 15)
----------------------------------------------------------------------
win_rate_last_year                  369
avg_player_rating                   334
possession_dominance                333
passing_accuracy                    332
fifa_points                         312
possession_avg                      306
climate_similarity_score            292
goal_efficiency                     283
shots_on_target_ratio               278
star_power                          270
shots_per_game                      264
travel_distance_avg                 262
goals_scored_avg                    254
defensive_solidity                  249
form_consistency                    232

======================================================================

## Cross-Validation Results
- XGBoost AUC-ROC: 0.6472 ± 0.0337
- Random Forest AUC-ROC: 0.6931 ± 0.0428
- Ensemble AUC-ROC: 0.6644 ± 0.0371

## Test Set Predictions
- Predictions shape: (250,)
- Probabilities shape: (250,)
- Min probability: 0.2308
- Max probability: 0.6897
- Mean probability: 0.4848
