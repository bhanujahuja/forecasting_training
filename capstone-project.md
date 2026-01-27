# Capstone Project: End-to-End Forecasting Pipeline

**Duration:** 10-15 hours  
**Prerequisites:** Modules 0-6  
**Learning Level:** Expert

---

## Overview

The capstone project integrates **everything** you've learned into a production-ready forecasting system. You'll work with real or realistic data, implement multiple approaches, and deliver professional results.

### Project Goals

✅ **Technical:** Build forecasting models that work  
✅ **Professional:** Document decisions and results clearly  
✅ **Practical:** Create something deployable  
✅ **Comprehensive:** Use techniques from all modules  
✅ **Portfolio-Ready:** Demonstrate your expertise  

---

## How to Choose Your Dataset

### Recommended Datasets

**Easy (Good for First Capstone):**
- Airline Passengers (144 observations, clear seasonality)
- Air Quality data (available on Kaggle)
- Monthly electricity load (seasonal + trend)

**Medium (Realistic Complexity):**
- Retail sales (multiple SKUs, trends, promotions)
- Website traffic (seasonality, spikes, day-of-week patterns)
- Energy consumption (demand, weather dependencies)

**Hard (Advanced):**
- Stock prices (non-stationary, news impact)
- COVID-19 cases (change points, multiple waves)
- Weather forecasting (multivariate, complex dynamics)

**Sources:**
- Kaggle: https://www.kaggle.com/datasets (search "time series" or "forecasting")
- UCI ML Repository: https://archive.ics.uci.edu/ml/index.php
- Federal Reserve (FRED): https://fred.stlouisfed.org/
- Quandl: https://www.quandl.com/
- Yahoo Finance: Historical stock data
- Government databases: Weather, economic indicators, etc.

### Criteria for Good Dataset

✅ **Do Choose:**
- 200+ observations (preferably 500+)
- Clear time stamps (daily, weekly, monthly, hourly)
- Some business relevance (can explain why forecasting matters)
- Mix of patterns (trend, seasonality, or noise)

❌ **Avoid:**
- < 100 observations (too small for reliable models)
- Artificial/synthetic data (use real examples)
- Extremely sparse data (lots of missing values)

---

## Phase 1: Problem Definition & EDA (2-3 hours)

### 1.1 Problem Statement

Define your forecasting problem clearly:

```markdown
## Problem Definition

**What:** Forecast [variable] for [time period]
**Why:** Business impact: [revenue impact, cost savings, risk reduction]
**How:** Will be used for [inventory planning, budget allocation, capacity, etc.]
**Success Metric:** Forecast within [X%] MAPE or [Y] MAE
```

**Example:**
```
What: Forecast daily electricity load for next 7 days
Why: Optimize power generation (avoid expensive peak generation)
How: Used for generator scheduling and cost optimization
Success: Within 5% MAPE
```

### 1.2 Exploratory Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('your_data.csv')

# Basic info
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Date range: {df.iloc[0]['date']} to {df.iloc[-1]['date']}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Time series
axes[0].plot(df['date'], df['target'])
axes[0].set_title('Full Time Series')
axes[0].grid()

# Histogram
axes[1].hist(df['target'], bins=30, edgecolor='black')
axes[1].set_title('Distribution')

# Box plot by season (if applicable)
df['month'] = df['date'].dt.month
df.boxplot(column='target', by='month', ax=axes[2])
axes[2].set_title('By Month (Seasonality Check)')

plt.tight_layout()
plt.show()

# Stationarity
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(df['target'].dropna())
print(f"\nADF Test p-value: {adf_result[1]:.6f}")
print("Stationary" if adf_result[1] < 0.05 else "Non-stationary (needs differencing)")

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['target'], model='additive', period=12)
fig = decomposition.plot()
plt.show()
```

### 1.3 Key Findings Document

```markdown
## EDA Findings

**Data Quality:**
- Missing values: [X%]
- Outliers detected: [Y points]
- Data quality issues: [list]

**Patterns:**
- Trend: [Upward/Downward/None] (describe)
- Seasonality: [Yes/No] (period = [X])
- Volatility: [High/Low/Changing]

**Characteristics:**
- Stationary: [Yes/No] (ADF p-value = X)
- Mean: [X], Std: [Y]
- Min: [X], Max: [Y]

**Data Preparation Needed:**
- [ ] Handle missing values
- [ ] Remove/flag outliers
- [ ] Differencing for stationarity
- [ ] Log transformation (if needed)
- [ ] Feature scaling
```

---

## Phase 2: Baseline Models (2 hours)

### 2.1 Simple Baselines

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Baselines
baselines = {
    'Mean': np.full(len(test), train['target'].mean()),
    'Naive': np.full(len(test), train['target'].iloc[-1]),
    'Seasonal Naive (12)': df['target'][train_size-12:train_size].values
}

# Evaluate
baseline_results = {}
for name, pred in baselines.items():
    mae = mean_absolute_error(test['target'], pred)
    mape = mean_absolute_percentage_error(test['target'], pred)
    baseline_results[name] = {'MAE': mae, 'MAPE': mape}

print("Baseline Results:")
print(pd.DataFrame(baseline_results).T)
```

### 2.2 Statistical Methods

```python
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Auto ARIMA
arima_model = auto_arima(train['target'], stepwise=True, seasonal=False)
arima_pred = arima_model.predict(n_periods=len(test))

# Exponential Smoothing
es_model = ExponentialSmoothing(
    train['target'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()
es_pred = es_model.forecast(steps=len(test))

# Prophet
df_prophet = train[['date', 'target']].copy()
df_prophet.columns = ['ds', 'y']
prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet.fit(df_prophet)
future = prophet.make_future_dataframe(periods=len(test), freq='D')
prophet_forecast = prophet.predict(future)
prophet_pred = prophet_forecast.iloc[-len(test):]['yhat'].values

# Evaluate
stat_results = {}
for name, pred in [('ARIMA', arima_pred), ('ES', es_pred), ('Prophet', prophet_pred)]:
    mae = mean_absolute_error(test['target'], pred)
    mape = mean_absolute_percentage_error(test['target'], pred)
    stat_results[name] = {'MAE': mae, 'MAPE': mape}

print("\nStatistical Methods:")
print(pd.DataFrame(stat_results).T)
```

---

## Phase 3: Machine Learning Models (2-3 hours)

### 3.1 Feature Engineering

```python
# Create features
X = pd.DataFrame()
X['lag_1'] = df['target'].shift(1)
X['lag_7'] = df['target'].shift(7)
X['lag_12'] = df['target'].shift(12)
X['rolling_mean_7'] = df['target'].rolling(7).mean()
X['rolling_std_7'] = df['target'].rolling(7).std()

# Seasonal features
X['month'] = df['date'].dt.month
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

# Trend
X['trend'] = np.arange(len(X))

# Target
X['target'] = df['target']
X = X.dropna()
y = X['target'].shift(-1).dropna()
X = X[:-1]

# Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.drop('target', axis=1))
X_test_scaled = scaler.transform(X_test.drop('target', axis=1))
```

### 3.2 Build ML Models

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)

# XGBoost
xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=200)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
pred_xgb = xgb_model.predict(X_test_scaled)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
gb.fit(X_train_scaled, y_train)
pred_gb = gb.predict(X_test_scaled)

# Evaluate
ml_results = {}
for name, pred in [('RF', pred_rf), ('XGB', pred_xgb), ('GB', pred_gb)]:
    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    ml_results[name] = {'MAE': mae, 'MAPE': mape}

print("\nML Models:")
print(pd.DataFrame(ml_results).T)
```

---

## Phase 4: Deep Learning (2-3 hours)

### 4.1 LSTM Model

```python
import tensorflow as tf
from tensorflow import keras

def create_sequences(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Prepare data
series_scaled = scaler.fit_transform(df['target'].values.reshape(-1, 1)).flatten()
X_dl, y_dl = create_sequences(series_scaled, lookback=12)

# Split
split_idx = int(len(X_dl) * 0.7)
val_idx = int(len(X_dl) * 0.85)

X_train_dl = X_dl[:split_idx].reshape(split_idx, 12, 1)
y_train_dl = y_dl[:split_idx]
X_val_dl = X_dl[split_idx:val_idx].reshape(val_idx-split_idx, 12, 1)
y_val_dl = y_dl[split_idx:val_idx]
X_test_dl = X_dl[val_idx:].reshape(len(X_dl)-val_idx, 12, 1)
y_test_dl = y_dl[val_idx:]

# Build LSTM
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(12, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(
    X_train_dl, y_train_dl,
    validation_data=(X_val_dl, y_val_dl),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=0
)

# Predict
pred_lstm = model.predict(X_test_dl, verbose=0)
mae_lstm = mean_absolute_error(y_test_dl, pred_lstm)
print(f"LSTM MAE: {mae_lstm:.4f}")
```

---

## Phase 5: Model Comparison & Ensemble (1-2 hours)

### 5.1 Comprehensive Comparison

```python
# Collect all predictions (on same test set)
all_models = {
    'ARIMA': arima_pred[:len(y_test)],
    'Prophet': prophet_pred[:len(y_test)],
    'Random Forest': pred_rf,
    'XGBoost': pred_xgb,
    'LSTM': pred_lstm.flatten()[:len(y_test)]
}

# Evaluate all
final_results = {}
for name, pred in all_models.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred)
    final_results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Rank': 0
    }

# Rank models
results_df = pd.DataFrame(final_results).T
results_df = results_df.sort_values('MAE')
print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(results_df)
print("="*60)
```

### 5.2 Ensemble

```python
# Simple average
ensemble_simple = np.mean(list(all_models.values()), axis=0)
mae_ensemble_simple = mean_absolute_error(y_test, ensemble_simple)

# Weighted by inverse MAE
maes = np.array([mean_absolute_error(y_test, pred) for pred in all_models.values()])
weights = 1 / maes
weights = weights / weights.sum()

ensemble_weighted = np.average(
    list(all_models.values()),
    axis=0,
    weights=weights
)
mae_ensemble_weighted = mean_absolute_error(y_test, ensemble_weighted)

print(f"\nEnsemble Simple MAE: {mae_ensemble_simple:.4f}")
print(f"Ensemble Weighted MAE: {mae_ensemble_weighted:.4f}")
print(f"\nEnsemble vs Best Individual:")
print(f"Best Individual: {results_df.iloc[0].name} ({results_df.iloc[0]['MAE']:.4f})")
print(f"Improvement: {results_df.iloc[0]['MAE'] - mae_ensemble_weighted:.4f}")
```

### 5.3 Visualization

```python
# Plot all predictions
plt.figure(figsize=(16, 8))
plt.plot(y_test.values, label='Actual', linewidth=2, marker='o', color='black')

colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
for (name, pred), color in zip(all_models.items(), colors):
    plt.plot(pred, label=name, alpha=0.7, color=color)

plt.plot(ensemble_weighted, label='Ensemble', linewidth=2, linestyle='--', color='purple')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('All Model Predictions vs Actual')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Phase 6: Advanced Analytics (1-2 hours)

### 6.1 Residual Analysis

```python
# Use best model
best_pred = all_models[results_df.index[0]]
residuals = y_test.values - best_pred

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Time series
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].grid()

# Histogram
axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_title('Residual Distribution')

# ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 0])

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])

plt.tight_layout()
plt.show()

# Check for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box = acorr_ljungbox(residuals, lags=10)
print("Ljung-Box Test (should be > 0.05):")
print(ljung_box)
```

### 6.2 Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies in training data
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(df['target'].values.reshape(-1, 1)) == -1

plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['target'], label='Series')
plt.scatter(df['date'][anomalies], df['target'][anomalies], color='red', label='Anomalies', s=100)
plt.legend()
plt.title('Detected Anomalies')
plt.grid(alpha=0.3)
plt.show()
```

---

## Phase 7: Deployment & Documentation (1-2 hours)

### 7.1 Save Best Model

```python
import joblib

best_model_name = results_df.index[0]
if best_model_name == 'Random Forest':
    joblib.dump(rf, 'best_forecast_model.pkl')
elif best_model_name == 'XGBoost':
    joblib.dump(xgb_model, 'best_forecast_model.pkl')
    
joblib.dump(scaler, 'scaler.pkl')

print(f"Best model ({best_model_name}) saved to 'best_forecast_model.pkl'")
```

### 7.2 Create Prediction Function

```python
def make_forecast(new_data):
    """
    Make predictions with best model
    """
    # Load model and scaler
    model = joblib.load('best_forecast_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Engineer features
    features = engineer_features(new_data)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    return prediction

# Example
recent_data = df.tail(12)  # Last 12 observations
forecast = make_forecast(recent_data)
print(f"Next month forecast: {forecast}")
```

### 7.3 Final Report

```markdown
# Capstone Project Report

## Executive Summary
- **Problem:** [Your problem]
- **Best Model:** [Model name]
- **Accuracy:** [MAE/MAPE]
- **Business Impact:** [Expected value/savings]

## Methodology
1. Exploratory data analysis
2. Baseline models (naive, ARIMA, Prophet)
3. Machine learning models (RF, XGB, GB)
4. Deep learning (LSTM)
5. Ensemble approach

## Results
- Best single model: [Name] with [MAE] MAE
- Ensemble improvement: [X%]
- Key insights: [List 3-5 insights]

## Next Steps
- Deploy to production
- Monitor performance weekly
- Retrain monthly with new data
- Investigate features [X, Y] for improvement

## Files
- `best_forecast_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `forecast_utils.py` - Prediction functions
- `capstone_notebook.ipynb` - Full analysis
```

---

## Evaluation Rubric

### Technical (50%)
- ✅ Data exploration and preparation (10%)
- ✅ Multiple model approaches (15%)
- ✅ Proper validation methodology (10%)
- ✅ Code quality and organization (15%)

### Results (30%)
- ✅ Model accuracy and comparison (15%)
- ✅ Ensemble vs individual models (10%)
- ✅ Error analysis and insights (5%)

### Presentation (20%)
- ✅ Clear documentation (10%)
- ✅ Visualizations and plots (5%)
- ✅ Business interpretation (5%)

---

## Submission Checklist

- [ ] Problem statement clearly defined
- [ ] EDA with visualizations
- [ ] At least 5 different models tested
- [ ] Proper train-test-val split
- [ ] Results comparison table
- [ ] Best model documented
- [ ] Code is clean and commented
- [ ] Final report/presentation
- [ ] Model saved and reproducible
- [ ] 5+ hours of work documented

---

## Congratulations!

You've completed the comprehensive forecasting course. You now understand:

✅ **Fundamentals:** Time series concepts, EDA, problem definition  
✅ **Classical:** Naive, smoothing, regression, ARIMA, SARIMA, Prophet  
✅ **Machine Learning:** Feature engineering, RF, XGBoost, hyperparameter tuning  
✅ **Deep Learning:** LSTM, CNN, hybrid architectures  
✅ **Advanced:** Multivariate, anomalies, ensembles, deployment  

**You're ready to:**
- 📊 Solve real forecasting problems
- 🚀 Deploy models to production
- 📈 Communicate results to stakeholders
- 🏆 Compete in forecasting competitions
- 💼 Build forecasting systems for your organization

**Next steps:**
- Apply to new datasets
- Specialize in a domain (finance, energy, retail, etc.)
- Learn about MLOps for production systems
- Explore advanced methods (Transformers, Attention, etc.)

---

*End of Capstone Project: End-to-End Forecasting Pipeline*

**Your Forecasting Journey Begins Here! 🎉**
