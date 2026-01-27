# Module 6: Advanced Topics in Forecasting

**Duration:** 6-8 hours (including mini-project)  
**Prerequisites:** Modules 0-5  
**Learning Level:** Advanced

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Build** multivariate forecasting models with external variables
2. **Detect** anomalies and outliers in time series
3. **Identify** change points and structural breaks
4. **Create** ensemble forecasts combining multiple approaches
5. **Optimize** decisions using forecast outputs
6. **Deploy** models to production environments
7. **Monitor** forecast quality over time

---

## 6.1 Multivariate Forecasting

### Why Multiple Variables?

Many real problems involve multiple related series:

```
Sales Forecast depends on:
├─ Historical sales (time series)
├─ Price (external variable)
├─ Marketing spend (external variable)
├─ Seasonality
├─ Trend
└─ Customer satisfaction (exogenous)
```

### Feature Engineering with Exogenous Variables

```python
import pandas as pd
import numpy as np

# Create multivariate dataset
dates = pd.date_range('2020-01', periods=120, freq='MS')
df = pd.DataFrame({
    'date': dates,
    'sales': np.cumsum(np.random.randn(120)) + 100,
    'price': 50 + 5 * np.sin(np.arange(120) / 12) + np.random.randn(120),
    'marketing': 1000 + 200 * np.sin(np.arange(120) / 6) + np.random.randn(120) * 50
})

# Feature engineering
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_12'] = df['sales'].shift(12)
df['price_lag_1'] = df['price'].shift(1)
df['marketing_lag_1'] = df['marketing'].shift(1)
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Target
df['target'] = df['sales'].shift(-1)

df_clean = df.dropna()
print(df_clean.head())
```

### ARIMAX: ARIMA with External Variables

```python
from statsmodels.tsa.arima.model import ARIMA

# Split
train_size = int(len(df_clean) * 0.8)
train = df_clean[:train_size]
test = df_clean[train_size:]

# ARIMA with exogenous variables
model_arimax = ARIMA(
    train['sales'],
    exog=train[['price', 'marketing']],
    order=(1, 1, 1)
)

fitted = model_arimax.fit()
print(fitted.summary())

# Forecast
forecast = fitted.get_forecast(
    steps=len(test),
    exog=test[['price', 'marketing']]
)

forecast_values = forecast.predicted_mean
mae_arimax = np.mean(np.abs(test['sales'].values - forecast_values))
print(f"ARIMAX MAE: {mae_arimax:.2f}")
```

### Vector AutoRegression (VAR)

```python
from statsmodels.tsa.api import VAR

# VAR: Each variable predicted from all variables' history
df_var = df_clean[['sales', 'price', 'marketing']].copy()
train_var = df_var[:train_size]

model_var = VAR(train_var)
fitted_var = model_var.fit(maxlags=2, ic='aic')

print(f"AR order: {fitted_var.k_ar}")

# Forecast
forecast_var = fitted_var.forecast(train_var.values[-fitted_var.k_ar:], steps=len(test))
mae_var = np.mean(np.abs(test['sales'].values - forecast_var[:, 0]))
print(f"VAR MAE: {mae_var:.2f}")
```

### ML with Exogenous Variables

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Prepare features
X_cols = ['sales_lag_1', 'sales_lag_12', 'price_lag_1', 'marketing_lag_1',
          'month_sin', 'month_cos']
X_train = df_clean[X_cols][:train_size]
y_train = df_clean['target'][:train_size]
X_test = df_clean[X_cols][train_size:]
y_test = df_clean['target'][train_size:]

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

pred_rf = rf.predict(X_test_scaled)
mae_rf = np.mean(np.abs(y_test - pred_rf))
print(f"RF with Exog MAE: {mae_rf:.2f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

---

## 6.2 Anomaly Detection

### Statistical Methods

```python
def detect_anomalies_zscore(series, threshold=3):
    """
    Z-score based anomaly detection
    """
    mean = np.mean(series)
    std = np.std(series)
    z_scores = np.abs((series - mean) / std)
    anomalies = z_scores > threshold
    return anomalies, z_scores

# Example
series = np.concatenate([
    np.random.normal(100, 5, 100),
    [150, 200],  # Anomalies
    np.random.normal(100, 5, 98)
])

anomalies, z_scores = detect_anomalies_zscore(series, threshold=3)

plt.figure(figsize=(14, 6))
plt.plot(series, label='Series')
plt.scatter(np.where(anomalies)[0], series[anomalies], color='red', label='Anomalies', s=100)
plt.legend()
plt.title('Z-Score Anomaly Detection')
plt.grid(alpha=0.3)
plt.show()
```

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Isolation Forest (unsupervised anomaly detection)
iso_forest = IsolationForest(contamination=0.02, random_state=42)
anomaly_labels = iso_forest.fit_predict(series.reshape(-1, 1))
anomalies_if = anomaly_labels == -1

plt.figure(figsize=(14, 6))
plt.plot(series, label='Series')
plt.scatter(np.where(anomalies_if)[0], series[anomalies_if], color='red', label='Anomalies', s=100)
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.grid(alpha=0.3)
plt.show()
```

### LSTM Autoencoder for Anomaly Detection

```python
import tensorflow as tf
from tensorflow import keras

# Autoencoders learn normal patterns
# High reconstruction error = anomaly

def build_autoencoder(input_dim):
    model = keras.Sequential([
        # Encoder
        keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        # Decoder
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare data
X_normal = series[~anomalies].reshape(-1, 1)
autoencoder = build_autoencoder(1)
autoencoder.fit(X_normal, X_normal, epochs=50, verbose=0)

# Detect anomalies
reconstructions = autoencoder.predict(series.reshape(-1, 1), verbose=0)
mse = np.mean(np.square(series.reshape(-1, 1) - reconstructions), axis=1)
threshold_ae = np.percentile(mse, 95)
anomalies_ae = mse > threshold_ae

plt.figure(figsize=(14, 6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(threshold_ae, color='red', linestyle='--', label='Threshold')
plt.scatter(np.where(anomalies_ae)[0], mse[anomalies_ae], color='red', s=100)
plt.legend()
plt.title('LSTM Autoencoder Anomaly Detection')
plt.grid(alpha=0.3)
plt.show()
```

---

## 6.3 Change Point Detection

### Structural Breaks in Time Series

```python
from ruptures import Pelt

# Generate series with change points
np.random.seed(42)
series_cp = np.concatenate([
    np.random.normal(100, 5, 50),
    np.random.normal(120, 5, 50),  # Level shift
    np.random.normal(90, 10, 50)   # Both level and variance change
])

# Detect change points
algo = Pelt(model="l2").fit(series_cp)
change_points = algo.predict(pen=10)

plt.figure(figsize=(14, 6))
plt.plot(series_cp, label='Series')
for cp in change_points:
    plt.axvline(cp, color='red', linestyle='--', label='Change Point')
plt.legend()
plt.title('Change Point Detection')
plt.grid(alpha=0.3)
plt.show()

print(f"Change points at indices: {change_points}")
```

---

## 6.4 Ensemble Forecasting

### Combining Multiple Models

```python
# Train multiple models
def build_ensemble_forecast(series, test_size=0.2):
    train_size = int(len(series) * (1 - test_size))
    train = series[:train_size]
    test = series[train_size:]
    
    # Model 1: ARIMA
    from pmdarima import auto_arima
    arima = auto_arima(train, stepwise=True)
    pred_arima = arima.predict(n_periods=len(test))
    
    # Model 2: Exponential Smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
    pred_es = es.forecast(steps=len(test))
    
    # Model 3: SARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    pred_sarima = sarima.forecast(steps=len(test))
    
    # Ensemble: Simple average
    ensemble_pred = (pred_arima + pred_es.values + pred_sarima) / 3
    
    # Ensemble: Weighted (by past performance)
    mae_arima = np.mean(np.abs(train[-12:] - pred_arima[-12:]))
    mae_es = np.mean(np.abs(train[-12:] - pred_es.values[-12:]))
    mae_sarima = np.mean(np.abs(train[-12:] - pred_sarima[-12:]))
    
    # Inverse weights (lower MAE = higher weight)
    weights = np.array([1/mae_arima, 1/mae_es, 1/mae_sarima])
    weights = weights / weights.sum()
    
    weighted_ensemble = (weights[0] * pred_arima + 
                        weights[1] * pred_es.values + 
                        weights[2] * pred_sarima)
    
    return {
        'ARIMA': pred_arima,
        'ExponentialSmoothing': pred_es.values,
        'SARIMA': pred_sarima,
        'Simple Ensemble': ensemble_pred,
        'Weighted Ensemble': weighted_ensemble
    }, test

predictions, test = build_ensemble_forecast(series)

# Compare
print("\nEnsemble Comparison:")
for name, pred in predictions.items():
    mae = np.mean(np.abs(test - pred))
    print(f"{name}: MAE = {mae:.2f}")
```

---

## 6.5 Probabilistic Forecasting

### Forecast Intervals and Uncertainty

```python
# Get confidence intervals from models
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
forecast_result = model_hw.get_forecast(steps=len(test))
forecast_df = forecast_result.summary_frame()

# forecast_df includes: predicted_mean, lower_ci, upper_ci

plt.figure(figsize=(14, 6))
plt.plot(test.index, test.values, label='Actual', linewidth=2, marker='o')
plt.plot(forecast_df.index, forecast_df['predicted_mean'], label='Forecast', linestyle='--')
plt.fill_between(
    forecast_df.index,
    forecast_df['mean_ci_lower'],
    forecast_df['mean_ci_upper'],
    alpha=0.3,
    label='95% Confidence Interval'
)
plt.legend()
plt.title('Forecast with Uncertainty Intervals')
plt.grid(alpha=0.3)
plt.show()
```

---

## 6.6 Production Deployment

### Model Serialization

```python
import pickle
import joblib

# Save model
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'forecast_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model
loaded_model = joblib.load('forecast_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Make prediction
new_data = [[...]]  # New features
new_data_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)
```

### Model Monitoring

```python
def monitor_forecast_quality(actual, predicted, window=20):
    """
    Track MAE over time to detect model degradation
    """
    rolling_errors = []
    for i in range(len(actual) - window):
        window_mae = np.mean(np.abs(
            actual[i:i+window] - predicted[i:i+window]
        ))
        rolling_errors.append(window_mae)
    
    return rolling_errors

rolling_mae = monitor_forecast_quality(test_actual, test_pred)

plt.figure(figsize=(14, 6))
plt.plot(rolling_mae, label='Rolling MAE (20-step window)')
plt.axhline(np.mean(rolling_mae), color='red', linestyle='--', label='Average MAE')
plt.xlabel('Time')
plt.ylabel('MAE')
plt.title('Forecast Quality Monitoring')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Alert if MAE increases
if rolling_mae[-1] > np.mean(rolling_mae) * 1.2:
    print("⚠️ Alert: Model performance degrading! Retrain recommended.")
```

---

## 6.7 Mini Project: Integrated Advanced Forecasting

```python
# Complete advanced forecasting pipeline

# 1. Load multivariate data
df = load_sales_data()  # sales, price, marketing, seasonality

# 2. Feature engineering
features = engineer_multivariate_features(df)

# 3. Split
train, test = train_test_split_timeseries(features, test_size=0.2)

# 4. Anomaly detection & handling
anomalies = detect_anomalies_isolation_forest(train['sales'])
train_clean = train[~anomalies]

# 5. Build models
models = {
    'ARIMAX': build_arimax(train_clean),
    'ML_RandomForest': build_rf_model(train_clean),
    'DeepLearning_LSTM': build_lstm_model(train_clean)
}

# 6. Ensemble
predictions = {name: model.predict(test) for name, model in models.items()}
ensemble_pred = np.mean(list(predictions.values()), axis=0)

# 7. Evaluate
results = {}
for name, pred in predictions.items():
    results[name] = {
        'MAE': mean_absolute_error(test['target'], pred),
        'RMSE': np.sqrt(mean_squared_error(test['target'], pred))
    }
results['Ensemble'] = {
    'MAE': mean_absolute_error(test['target'], ensemble_pred),
    'RMSE': np.sqrt(mean_squared_error(test['target'], ensemble_pred))
}

print(pd.DataFrame(results).T)

# 8. Deploy best model
best_model = models['ML_RandomForest']
joblib.dump(best_model, 'production_model.pkl')
joblib.dump(scaler, 'production_scaler.pkl')

# 9. Monitor
future_mae = monitor_forecast_quality(test['target'], ensemble_pred)
```

---

## 6.8 Key Takeaways

✅ **Advanced Techniques:**
- Use exogenous variables for better predictions
- Detect and handle anomalies appropriately
- Combine models for robustness
- Include uncertainty in forecasts
- Monitor model quality in production

❌ **Common Pitfalls:**
- Ignoring external variables
- Deleting anomalies without investigation
- Not monitoring for performance degradation
- Assuming model works forever without retraining
- Overfitting ensemble weights to test data

---

## Progress Checkpoint

**Completion: ~80%** ✓

You now understand:
- Multivariate forecasting with external variables
- Anomaly detection and handling
- Change point detection
- Ensemble methods for robustness
- Production deployment and monitoring
- Probabilistic forecasting with intervals

**Next:** Capstone Project
- Integrate everything into one comprehensive project
- Real-world data and business problem
- Full pipeline from exploration to deployment

**Time Estimate for Module 6:** 6-8 hours  
**Estimated Combined Progress:** 80% → 90%

---

*End of Module 6: Advanced Topics in Forecasting*
