# Module 4: Machine Learning for Forecasting

**Duration:** 8-10 hours (including mini-project)  
**Prerequisites:** Modules 0-3  
**Learning Level:** Intermediate to Advanced

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Convert** time series into supervised learning problems
2. **Engineer** time-aware features for ML models
3. **Build** Random Forest and Gradient Boosting models
4. **Apply** proper time-series cross-validation
5. **Tune** hyperparameters systematically
6. **Interpret** feature importance and model decisions
7. **Compare** ML models vs statistical methods

---

## Why Machine Learning for Forecasting?

### ML vs Statistical Methods

| Aspect | ARIMA/Prophet | Machine Learning |
|--------|---------------|------------------|
| **Flexibility** | Fixed patterns | Learns complex patterns |
| **Non-linearity** | Limited | Excellent |
| **Multiple features** | Difficult | Natural |
| **Large datasets** | Slow | Fast |
| **Interpretability** | High | Medium |
| **Hyperparameters** | Few | Many |

### When to Use ML

- ✅ Multiple input features (price, marketing, weather, etc.)
- ✅ Non-linear relationships
- ✅ Large datasets (1000+ observations)
- ✅ Competitive forecasting (Kaggle, research)
- ❌ Not for: Explainability-critical business decisions (use ARIMA/Prophet)

---

## 4.1 Feature Engineering for Time Series

### Converting Time Series → Supervised Learning

The key insight: Create a **feature matrix** from historical values.

```
Original Series:    [10, 12, 14, 13, 15, 18]

Target (y):         [14, 13, 15, 18]
Features (X):       
  - lag_1: [12, 14, 13, 15]
  - lag_2: [10, 12, 14, 13]
  - lag_3: [10, 12, 14]
```

### Lag Features

```python
import pandas as pd
import numpy as np

def create_lag_features(series, lags=[1, 2, 3]):
    """Create lagged features from series"""
    df = pd.DataFrame({'y': series})
    
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    df['target'] = df['y'].shift(-1)  # Next period
    
    return df.dropna()

# Example
series = [10, 12, 14, 13, 15, 18, 20]
features_df = create_lag_features(series, lags=[1, 2, 3])
print(features_df)

# Output:
#      y  lag_1  lag_2  lag_3  target
# 3   13     14     12     10      15
# 4   15     13     14     12      18
# 5   18     15     13     14      20
```

### Rolling Statistics

```python
def create_rolling_features(series, windows=[3, 6]):
    """Add rolling mean, std, min, max"""
    df = pd.DataFrame({'y': series})
    
    for w in windows:
        df[f'rolling_mean_{w}'] = df['y'].rolling(w).mean()
        df[f'rolling_std_{w}'] = df['y'].rolling(w).std()
        df[f'rolling_min_{w}'] = df['y'].rolling(w).min()
        df[f'rolling_max_{w}'] = df['y'].rolling(w).max()
    
    df['target'] = df['y'].shift(-1)
    return df.dropna()

features_df = create_rolling_features(series, windows=[3])
```

### Seasonal Features

```python
def add_seasonal_features(dates):
    """Add month, quarter, year, and cyclical encoding"""
    df = pd.DataFrame({'date': dates})
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Cyclical encoding (sine/cosine) handles month circularity
    # Jan=1 and Dec=12 should be close, not far apart
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

dates = pd.date_range('2020-01', periods=24, freq='MS')
seasonal_features = add_seasonal_features(dates)
print(seasonal_features.head())
```

### Complete Feature Engineering Pipeline

```python
def engineer_all_features(series, dates, lags=[1, 6, 12], rolling_windows=[3, 12]):
    """
    Complete feature engineering
    """
    # Start with lags
    df = pd.DataFrame({'y': series, 'date': dates})
    
    # Add lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # Add rolling features
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df['y'].rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].rolling(w).std()
    
    # Add seasonal features
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add trend feature
    df['time_idx'] = np.arange(len(df))
    
    # Target
    df['target'] = df['y'].shift(-1)
    
    return df.dropna()

# Usage
dates = pd.date_range('2020-01', periods=120, freq='MS')
series = np.random.randn(120).cumsum() + 100
features = engineer_all_features(series, dates)

print(f"Features created: {features.shape}")
print(features.head())
```

---

## 4.2 Time-Series Train-Test Split

### Never Shuffle Time Series!

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ❌ WRONG - Random shuffle breaks temporal order
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ CORRECT - Sequential split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale features (fit on train, apply to test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Time-Series Cross-Validation

```python
# Expanding window: train set grows over time
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    
    # Train and evaluate
    model.fit(X_tr_scaled, y_tr)
    pred = model.predict(X_val_scaled)
    
    mae = mean_absolute_error(y_val, pred)
    cv_scores.append(mae)
    
    print(f"Fold {fold + 1}: MAE = {mae:.2f}")

print(f"\nAverage CV MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

---

## 4.3 ML Model Implementations

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Build model
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Tree depth
    min_samples_split=5,   # Min samples to split
    random_state=42
)

# Train
rf_model.fit(X_train_scaled, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, MAPE: {mape_rf:.2f}%")

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

### XGBoost: Extreme Gradient Boosting

```python
import xgboost as xgb

# Build model
xgb_model = xgb.XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train with early stopping
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    early_stopping_rounds=10,
    verbose=False
)

# Predict
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost - MAE: {mae_xgb:.2f}")

# Plot feature importance
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```

### LightGBM: Fast Gradient Boosting

```python
import lightgbm as lgb

# Build model (similar to XGBoost, but faster)
lgb_model = lgb.LGBMRegressor(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    num_leaves=31,
    random_state=42
)

lgb_model.fit(X_train_scaled, y_train)
y_pred_lgb = lgb_model.predict(X_test_scaled)

mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
print(f"LightGBM - MAE: {mae_lgb:.2f}")
```

---

## 4.4 Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Grid search
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

# Use best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_best):.2f}")
```

### Random Search (More Efficient)

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=20,  # Try 20 combinations
    cv=TimeSeriesSplit(n_splits=3),
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

---

## 4.5 Model Comparison

```python
# Collect all models
models = {
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb,
    'LightGBM': y_pred_lgb
}

# Evaluate all
comparison = {}
for name, predictions in models.items():
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    comparison[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

results = pd.DataFrame(comparison).T.sort_values('MAE')
print("\nML Models Comparison:")
print(results)

# Visualize predictions
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Actual', linewidth=2, marker='o')
for name, predictions in models.items():
    plt.plot(predictions, label=name, alpha=0.7)
plt.legend()
plt.title('ML Model Predictions vs Actual')
plt.grid(alpha=0.3)
plt.show()
```

---

## 4.6 Mini Project: ML Forecasting Pipeline

### Complete Project Code

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

# Step 1: Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
series = df['Passengers'].values
dates = pd.to_datetime(df['Month'])

# Step 2: Feature engineering
X = pd.DataFrame()
X['lag_1'] = pd.Series(series).shift(1)
X['lag_6'] = pd.Series(series).shift(6)
X['lag_12'] = pd.Series(series).shift(12)
X['rolling_mean_3'] = pd.Series(series).rolling(3).mean()
X['rolling_mean_12'] = pd.Series(series).rolling(12).mean()
X['month'] = dates.dt.month
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
X['trend'] = np.arange(len(series))
X['y'] = series

X = X.dropna()
y = X['y'].shift(-1).dropna()
X = X[:-1]

# Step 3: Train-test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Drop target from features
X_train = X_train.drop('y', axis=1)
X_test = X_test.drop('y', axis=1)

# Step 4: Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build models
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
xgb_m = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42)

rf.fit(X_train_scaled, y_train)
xgb_m.fit(X_train_scaled, y_train)

# Step 6: Predict
y_pred_rf = rf.predict(X_test_scaled)
y_pred_xgb = xgb_m.predict(X_test_scaled)

# Step 7: Evaluate
print(f"RF MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"XGB MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")

# Step 8: Cross-validation
tscv = TimeSeriesSplit(n_splits=3)
rf_cv_scores = []
for train_idx, val_idx in tscv.split(X_train_scaled):
    X_tr = X_train_scaled[train_idx]
    X_val = X_train_scaled[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]
    
    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_cv.fit(X_tr, y_tr)
    pred = rf_cv.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    rf_cv_scores.append(mae)

print(f"RF CV MAE: {np.mean(rf_cv_scores):.2f} ± {np.std(rf_cv_scores):.2f}")

# Step 9: Feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
print(importance_df.head())
```

---

## 4.7 Key Insights

### When ML Beats Statistical Methods
- Many input features (multi-variable)
- Non-linear patterns
- Large datasets
- Competitive environments

### When Statistical Methods Win
- Few features
- Limited data
- Interpretability required
- Stable patterns

### Best Practices
- Always use time-series CV (not random shuffle)
- Engineer thoughtful features (lags, rolling, seasonal)
- Scale features before training
- Compare multiple algorithms
- Validate on completely separate test set

---

## Progress Checkpoint

**Completion: ~50%** ✓

You now understand:
- Feature engineering for time series
- Random Forest, XGBoost, LightGBM
- Time-series cross-validation
- Hyperparameter tuning
- Feature importance analysis

**Next Module (Module 5):** Deep Learning Methods
- LSTMs for sequence modeling
- 1D CNNs for fast forecasting
- Attention mechanisms
- Multi-step ahead forecasting

**Time Estimate for Module 4:** 8-10 hours  
**Estimated Combined Progress:** 50% → 65%

---

*End of Module 4: Machine Learning for Forecasting*
