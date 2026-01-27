# Deep Dive: Critical Concepts Explained

This document provides in-depth explanations of complex concepts with visual aids and worked examples.

---

## 1. Stationarity vs Non-Stationarity

### What is Stationarity?

**Simple Definition:** A stationary time series has constant statistical properties (mean, variance, autocorrelation) over time.

**Visual Comparison:**

```
Non-Stationary (has trend):        Stationary (no trend):
     |╱╱╱╱╱╱╱╱╱╱╱                       |┌─┐  ┌─┐  ┌─┐
     |╱╱╱╱╱╱╱╱╱╱╱                       |└─┘  └─┘  └─┘
     |╱╱╱╱╱╱╱╱╱╱╱                       |
     |────────────                       |────────────
     Time ────→                          Time ────→

Mean keeps changing               Mean stays constant
```

### Why It Matters

**For ARIMA Models:**
- ARIMA assumes stationarity
- If data is non-stationary, ARIMA produces unreliable forecasts
- Solution: Difference the data (subtract previous value) to make it stationary

### How to Test: Augmented Dickey-Fuller (ADF) Test

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(data)
print(f"p-value: {result[1]}")

if result[1] < 0.05:
    print("✓ Data is STATIONARY (reject null hypothesis)")
    print("→ Can use ARIMA directly")
else:
    print("✗ Data is NON-STATIONARY (fail to reject)")
    print("→ Need to difference the data")
```

### Differencing Example

```
Original:     [10, 12, 14, 16, 18, 20]  ← Clearly a trend
Differenced:  [2, 2, 2, 2, 2]            ← Constant! (stationary)
```

```python
import pandas as pd

# Original non-stationary
original = pd.Series([10, 12, 14, 16, 18, 20])

# Difference it
differenced = original.diff().dropna()  # [2, 2, 2, 2, 2]

# Test differenced data
adf_result = adfuller(differenced)
# Should have p-value < 0.05 (stationary)
```

---

## 2. ACF vs PACF Plots

### What They Show

**ACF (Autocorrelation Function):**
- Shows correlation between a value and its past values
- Answers: "Does the value depend on previous values?"

**PACF (Partial Autocorrelation Function):**
- Shows correlation removing intermediate effects
- Answers: "Does lag-3 matter after accounting for lag-1 and lag-2?"

### Reading ACF/PACF Plots

```
Strong positive correlation:        No correlation:
    |█████                              |   
    |  █                                |
    |  █                                |
    |  ░                                |
    |  ░                                |
    |──────────                         |──────────
    Lag                                 Lag

Past strongly                      Past doesn't
predicts future                    predict future
```

### Using ACF/PACF to Choose ARIMA(p,d,q)

```
ACF behavior          PACF behavior         → Suggested ARIMA
─────────────────────────────────────────────────────────────
Decays slowly         Cuts off at lag p     → (p, d, 0)
Cuts off at lag q     Decays slowly         → (0, d, q)
Both decay            Both decay            → (p, d, q)
─────────────────────────────────────────────────────────────

Example:
If ACF cuts off at lag 2 and PACF decays slowly
→ Try ARIMA(0, d, 2)  (moving average with q=2)
```

### Worked Example

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Create a sample dataset
data = [10, 11, 13, 12, 14, 15, 17, 16, 18, 19]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot ACF
plot_acf(data, lags=5, ax=axes[0])
axes[0].set_title('ACF Plot')

# Plot PACF
plot_pacf(data, lags=5, ax=axes[1])
axes[1].set_title('PACF Plot')

plt.tight_layout()
plt.show()

# Interpretation:
# - If ACF decays slowly → Need differencing or autoregressive terms
# - If PACF cuts off → AR component present
# - If ACF cuts off → MA component present
```

---

## 3. Feature Engineering for Time Series

### Why Features Matter

```
Bad features:  [1, 2, 3, 4, 5] → Model: "Just count up!"
Good features: [
    value: 1,
    lag1: 0,
    rolling_mean_3: 0,
    day_of_week: 1,
    is_holiday: 0,
    trend: -0.1
] → Model: "These features predict the outcome!"
```

### Essential Feature Types

#### 1. Lag Features (Past Values)

```python
data = pd.Series([10, 12, 14, 16, 18, 20])

# Create lags
lags = {}
for lag in [1, 2, 7]:
    lags[f'lag_{lag}'] = data.shift(lag)

# Result:
#    value  lag_1  lag_2  lag_7
# 0     10    NaN    NaN    NaN
# 1     12   10.0    NaN    NaN
# 2     14   12.0   10.0    NaN
# 3     16   14.0   12.0    NaN
```

#### 2. Rolling Statistics

```python
# Rolling mean smooths noise
rolling_mean_3 = data.rolling(3).mean()
# [NaN, NaN, 12.0, 14.0, 16.0, 18.0]  (average of 3 values)

# Rolling std captures volatility
rolling_std_7 = data.rolling(7).std()
# Higher std = more volatility
```

#### 3. Trend Features

```python
# Linear trend: increasing counter
data['trend'] = np.arange(len(data))

# Polynomial trend: capture non-linear growth
data['trend_squared'] = np.arange(len(data)) ** 2
```

#### 4. Seasonal Features (Using Sin/Cos)

```python
import numpy as np

# For monthly data with yearly seasonality
n_months = 12
data['month_sin'] = np.sin(2 * np.pi * data['month'] / n_months)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / n_months)

# Why sine/cosine? Because month 11 is close to month 12 and month 1
# Numeric difference: 11 to 1 is 10 (far!)
# But sine/cosine: very close values (correct!)
```

#### 5. External Features

```python
# If you have external data, use it!
data['is_holiday'] = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1]  # Holiday indicators
data['temperature'] = [65, 68, 72, 70, 75, 78, 77, 76, 74, 71]  # Weather
data['is_weekend'] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]  # Day of week
```

### Feature Engineering Checklist

```python
def engineer_features(df):
    """Create features for time series ML."""
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['target'].shift(lag)
    
    # Rolling features
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = df['target'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['target'].rolling(window).std()
    
    # Trend
    df['trend'] = np.arange(len(df))
    
    # Seasonality
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    
    # Sin/Cos encoding for cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df
```

---

## 4. Time Series Cross-Validation

### Why Random Shuffle is Wrong

```
❌ WRONG: Random shuffling
Original: [1, 2, 3, 4, 5, 6, 7, 8]
Shuffled: [3, 7, 1, 5, 2, 8, 4, 6]
Split:    [3,7,1] Train | [5,2,8,4,6] Test
Problem:  Test has 5 (future) but training has 6,7,8 (even further future!)
          → Model sees future data during training
          → Overly optimistic accuracy

✅ RIGHT: Time series split
Original: [1, 2, 3, 4, 5, 6, 7, 8]
Split 1:  [1] Train | [2,3,4,5,6,7,8] Test
Split 2:  [1,2] Train | [3,4,5,6,7,8] Test
Split 3:  [1,2,3] Train | [4,5,6,7,8] Test
...
Result:   Model only predicts future from past
          → Realistic accuracy
```

### Time Series Cross-Validation Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

# Create 5 time-based splits
tscv = TimeSeriesSplit(n_splits=5)

# Split the data
for split_num, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Split {split_num + 1}:")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Fit and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  Score: {score:.3f}\n")

# Average the scores
print(f"Average CV Score: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

---

## 5. LSTM (Long Short-Term Memory) Networks

### The Problem LSTM Solves

```
Regular Neural Network:
Input: [1, 2, 3, 4, 5]
       ↓↓↓↓↓
    [Dense Layer] → Predicts 6
Problem: Only considers immediate pattern, forgets context

LSTM Network:
Input: [1, 2, 3, 4, 5]
       ↓↓↓↓↓
    [LSTM Cell with Memory] → Remembers: trend is +1 each step
       ↓
    Predicts 6, 7, 8, ... correctly!
```

### LSTM Architecture (Simplified)

```
Cell State (Memory):  ─────[┌──────┐]─────
                            │Memory│
                            │Input │
                            │Output│
                            └──────┘

Input Gate:     "Should we add new information?"  [0=No, 1=Yes]
Forget Gate:    "Should we forget old information?" [0=No, 1=Yes]
Output Gate:    "What information should we output?" [0=None, 1=All]

Each gate learns what to do based on training data.
```

### Why LSTM > Feedforward for Time Series

```
Feedforward Neural Network:
x[t] → [Dense] → output[t]
(Only uses current input)

LSTM:
x[t-1], x[t] → [LSTM] → Maintains internal state → output[t]
(Remembers past, can capture long-term dependencies)
```

### Sequence Preparation for LSTM

```python
def create_sequences(data, lookback=10):
    """Convert 1D series to 3D sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback):
        # Past 10 values
        X.append(data[i:i+lookback])
        # Value to predict
        y.append(data[i+lookback])
    
    return np.array(X), np.array(y)

# Example
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
X, y = create_sequences(data, lookback=3)

# X shape: (N, lookback, 1) = (9, 3, 1)
# [1, 2, 3] → 4
# [2, 3, 4] → 5
# [3, 4, 5] → 6
# ...
```

---

## 6. Ensemble Methods

### Why Combine Models?

```
Model A: Predicts [10, 11, 12, 13, 14]
Model B: Predicts [9, 10, 11, 12, 13]
Model C: Predicts [11, 12, 13, 14, 15]
Actual:  [10, 11, 12, 13, 14]

Individual errors: [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]

Ensemble (Average): [(10+9+11)/3, (11+10+12)/3, ...]
                  = [10.0, 11.0, 12.0, 13.0, 14.0]
                  Error: [0, 0, 0, 0, 0] ✓ Perfect!
```

### Why It Works

1. **Diversification:** Different models make different errors
2. **Error Cancellation:** Overestimates and underestimates cancel out
3. **Robustness:** If one model fails, others compensate

### Ensemble Methods

#### Simple Averaging
```python
ensemble = (model_a + model_b + model_c) / 3
```
**Best for:** Similar-performing models

#### Weighted Averaging
```python
ensemble = 0.5 * model_a + 0.3 * model_b + 0.2 * model_c
```
**Best for:** Different model quality levels

#### Voting
```python
# For classification, vote on class
predictions = [model_a, model_b, model_c]
ensemble = np.bincount(predictions).argmax()
```

#### Stacking
```python
# Train meta-model to learn optimal combination
meta_model = LinearRegression()
meta_inputs = np.array([model_a, model_b, model_c]).T
meta_model.fit(meta_inputs, y_actual)
ensemble = meta_model.predict(meta_inputs)
```

### Ensemble Checklist

- [ ] Pick 3-5 diverse models (statistical + ML + DL)
- [ ] Train each on same train set
- [ ] Evaluate on same test set
- [ ] Combine predictions (average, weighted, or stack)
- [ ] Compare ensemble vs individual models
- [ ] Use test set only for final comparison

---

## 7. Hyperparameter Tuning

### What's a Hyperparameter?

```
Model Parameters (learned):
└─ Coefficients in linear regression: y = 0.5*x + 2.1
└─ Weights in neural networks
└─ Learned automatically during training

Hyperparameters (user-specified):
└─ Learning rate (how fast to update weights)
└─ Number of hidden units in neural network
└─ Regularization strength (L1/L2)
└─ Number of trees in random forest
└─ Must be specified before training
```

### Tuning Strategy

```
1. Start with default values
   └─ Often reasonable defaults exist

2. Do Grid Search (try specific values)
   └─ XGBoost: max_depth in [3, 5, 7, 9, 11]
   └─ Try all combinations

3. Narrow down to best region
   └─ If 7 was best, try [5, 6, 7, 8, 9]

4. Fine-tune carefully
   └─ Diminishing returns as you optimize

5. Validate on held-out test set
   └─ Ensure good performance on unseen data
```

### Common Hyperparameters by Model

**Linear Regression:**
- Regularization type (L1, L2, none)
- Regularization strength (C or alpha)

**Random Forest:**
- n_estimators (number of trees): 50-500
- max_depth (tree depth): 3-15
- min_samples_split: 2-10
- max_features: 'sqrt', 'log2', or number

**XGBoost:**
- learning_rate: 0.01-0.3 (lower = more careful)
- max_depth: 3-8
- subsample: 0.6-1.0 (row sampling)
- colsample_bytree: 0.6-1.0 (column sampling)
- reg_alpha, reg_lambda: Regularization

**LSTM:**
- Number of units: 32-256
- Dropout rate: 0.2-0.5
- Learning rate: 0.001-0.01
- Batch size: 16-128

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Create model
rf = RandomForestRegressor()

# Grid search with time-series CV
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1  # Use all cores
)

# Fit (searches all combinations)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
```

---

## 8. Metrics Deep Dive

### When to Use Each Metric

```
MAE (Mean Absolute Error):
- Interpretable (same units as data)
- Fair to all errors (doesn't penalize outliers)
- Use when: All errors are equally important

RMSE (Root Mean Squared Error):
- Penalizes large errors heavily
- Use when: Large errors are especially bad
- Sensitive to outliers

MAPE (Mean Absolute Percentage Error):
- Percentage-based (compare across different scales)
- Use when: Comparing multiple series with different magnitudes
- Problem: Undefined when actual value is 0

R² (Coefficient of Determination):
- Proportion of variance explained (0-1)
- 1.0 = perfect, 0.0 = no better than mean
- Use when: Understanding model quality relative to baseline
```

### Worked Examples

```python
# Actual vs Predicted
actual = [10, 12, 14, 16, 18]
pred   = [11, 11, 15, 15, 18]  # Over, under, over, under, perfect

# MAE: Average of absolute errors
errors = [|10-11|, |12-11|, |14-15|, |16-15|, |18-18|]
       = [1, 1, 1, 1, 0]
MAE = (1+1+1+1+0)/5 = 0.8

# RMSE: Penalizes large errors
squared_errors = [1², 1², 1², 1², 0²] = [1, 1, 1, 1, 0]
RMSE = √(1+1+1+1+0)/5 = √(4/5) = 0.894

# MAPE: Percentage
pct_errors = [|1|/10, |1|/12, |1|/14, |1|/16, |0|/18]
MAPE = average(pct_errors) * 100 ≈ 7.5%
```

---

## Summary

These are the most critical concepts for successful forecasting:

1. **Stationarity** - Required for ARIMA
2. **ACF/PACF** - How to identify ARIMA parameters
3. **Features** - Make or break ML models
4. **Time Series CV** - Proper validation
5. **LSTM** - When deep learning wins
6. **Ensembles** - Combine strengths
7. **Hyperparameters** - Optimize model performance
8. **Metrics** - Measure what matters

Master these, and you'll excel at forecasting!

---

*Last Updated: January 28, 2026*
