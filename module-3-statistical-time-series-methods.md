# Module 3: Statistical Time Series Methods

**Duration:** 8-10 hours (including mini-project)  
**Prerequisites:** Modules 0-2  
**Learning Level:** Intermediate

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand** stationarity and why it matters for time series
2. **Diagnose** non-stationary data and apply differencing
3. **Interpret** ACF and PACF plots to identify ARIMA parameters
4. **Build** and evaluate ARIMA and SARIMA models
5. **Use** Facebook's Prophet for trend and seasonality
6. **Compare** statistical models and choose the best approach
7. **Deploy** forecasts with confidence intervals

---

## Why This Module Matters

### Statistical Methods vs. Baselines

Module 2 taught us simple methods. Now we learn **intelligent** methods:

```
Naive Method:       "Tomorrow = Today"
Statistical Method: "Tomorrow ≈ Today + (trend) + (seasonal) + (error pattern)"
```

Statistical methods:
- ✅ Explain what patterns they capture
- ✅ Work on smaller datasets
- ✅ Provide confidence intervals
- ✅ Handle trend and seasonality automatically
- ✅ Foundation for understanding advanced methods

---

## 3.1 Stationarity: The Foundation Concept

### What is Stationarity?

A time series is **stationary** if:
1. **Constant mean** (no upward/downward drift)
2. **Constant variance** (volatility doesn't change)
3. **No seasonal patterns** (or seasonality is removed)
4. **Autocorrelation** depends only on lag, not time

### Visual Examples

```
STATIONARY (good for ARIMA):
    ___
   /   \    /   \    /   \
  /     \__/     \__/     

NON-STATIONARY (trends, changing mean):
    /         Increasing trend
   /
  /    
        /\        Changing variance
       /  \
      /    \____
```

### Why Stationarity Matters

ARIMA assumes stationarity. If data isn't stationary:
- Model may overfit historical trends
- Forecasts become unreliable
- Statistical properties change over time

### Testing for Stationarity

#### Augmented Dickey-Fuller (ADF) Test

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
series = df['Passengers'].values

# Perform ADF test
result = adfuller(series)

print("ADF Test Results:")
print(f"Test Statistic: {result[0]:.6f}")
print(f"P-value: {result[1]:.6f}")
print(f"Critical Values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.3f}")

# Interpretation
if result[1] <= 0.05:
    print("✓ Series IS stationary (p-value < 0.05)")
else:
    print("✗ Series is NOT stationary (p-value >= 0.05)")
    print("  → Need to apply differencing")
```

**How to interpret:**
- **P-value < 0.05:** Stationary ✓
- **P-value ≥ 0.05:** Non-stationary (needs differencing)

#### KPSS Test

```python
from statsmodels.tsa.stattools import kpss

result_kpss = kpss(series, regression='c')
print(f"KPSS Test P-value: {result_kpss[1]:.6f}")
# KPSS opposite: p-value < 0.05 means NON-stationary
```

### Making Data Stationary: Differencing

**Differencing:** Subtract each value from the previous value.

```
Original:    [10, 12, 14, 13, 15]
Differenced: [    2,  2, -1,  2]  (change from one period to next)
```

**Mathematical notation:** d=1 in ARIMA(p,d,q)

```python
# First differencing
differenced_1 = series.diff().dropna()

# Check if differenced data is stationary
result_diff = adfuller(differenced_1)
print(f"Differenced series p-value: {result_diff[1]:.6f}")

# Visualize difference
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(series, marker='o')
axes[0].set_title('Original Series (Non-Stationary)')
axes[0].set_ylabel('Value')
axes[0].grid()

axes[1].plot(differenced_1, marker='o', color='orange')
axes[1].set_title('First Differenced Series (Stationary)')
axes[1].set_ylabel('Change')
axes[1].grid()

plt.tight_layout()
plt.show()

# Sometimes need second differencing
if result_diff[1] > 0.05:
    differenced_2 = differenced_1.diff().dropna()
    result_diff2 = adfuller(differenced_2)
    print(f"Second differenced p-value: {result_diff2[1]:.6f}")
```

---

## 3.2 ACF and PACF: Reading the Plots

These plots tell you the ARIMA parameters (p, q).

### Autocorrelation Function (ACF)

Shows correlation between data and its lags.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# ACF plot
plot_acf(differenced_1, lags=40, ax=axes[0])
axes[0].set_title('ACF Plot')

# PACF plot
plot_pacf(differenced_1, lags=40, ax=axes[1])
axes[1].set_title('PACF Plot')

plt.tight_layout()
plt.show()
```

### How to Read Them

**ACF Pattern → MA Order (q):**
- Cuts off sharply after lag q → MA(q)
- Decays gradually → Higher MA order

**PACF Pattern → AR Order (p):**
- Cuts off sharply after lag p → AR(p)
- Decays gradually → Higher AR order

**Example:**
```
ACF: Sharp cutoff after lag 1 → MA(1)
PACF: Gradual decay → High AR order
Result: Try ARIMA(2, 1, 1)
```

---

## 3.3 ARIMA: AutoRegressive Integrated Moving Average

### ARIMA(p, d, q) Components

| Parameter | Meaning | Range |
|-----------|---------|-------|
| **p** | AR order (past values) | 0, 1, 2, ... |
| **d** | Differencing (make stationary) | 0, 1, 2 |
| **q** | MA order (past errors) | 0, 1, 2, ... |

### Building an ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Split data
train_size = int(len(df) * 0.8)
train, test = df['Passengers'][:train_size], df['Passengers'][train_size:]

# Fit ARIMA(1,1,1)
model = ARIMA(train, order=(1, 1, 1))
fitted_model = model.fit()

# Print summary
print(fitted_model.summary())

# Forecast
forecast = fitted_model.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast, label='ARIMA(1,1,1) Forecast', linestyle='--')
plt.legend()
plt.title('ARIMA Forecast')
plt.grid()
plt.show()
```

### Choosing Parameters with auto_arima

Instead of guessing, use automated parameter selection:

```python
from pmdarima import auto_arima

# Automatically find best parameters
auto_model = auto_arima(
    train,
    start_p=0, start_q=0, max_p=5, max_q=5,
    d=None,  # Let it auto-detect
    seasonal=False,
    stepwise=True,
    trace=True  # Show progress
)

print(auto_model.summary())

# Use the best model
auto_forecast = auto_model.predict(n_periods=len(test))
auto_mae = mean_absolute_error(test, auto_forecast)
print(f"Auto ARIMA MAE: {auto_mae:.2f}")
```

---

## 3.4 SARIMA: Seasonal ARIMA

### Adding Seasonality

**SARIMA(p,d,q)(P,D,Q,s):**
- (p,d,q) = Regular ARIMA
- (P,D,Q) = Seasonal ARIMA
- s = Season length (12 for monthly)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA with seasonality
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fitted = sarima_model.fit(disp=False)
sarima_forecast = sarima_fitted.forecast(steps=len(test))
sarima_mae = mean_absolute_error(test, sarima_forecast)

print(f"SARIMA MAE: {sarima_mae:.2f}")

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', linestyle='--')
plt.legend()
plt.title('SARIMA vs Actual')
plt.grid()
plt.show()
```

---

## 3.5 Prophet: Trend and Seasonality Decomposition

Facebook's Prophet is great for:
- Business data with strong seasonality
- Holiday effects
- Automatic changepoint detection
- Uncertainty intervals

```python
from prophet import Prophet

# Prepare data for Prophet
df_prophet = df.rename(columns={'Month': 'ds', 'Passengers': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet = df_prophet[['ds', 'y']]

# Split
train_prophet = df_prophet[:train_size]
test_prophet = df_prophet[train_size:]

# Fit Prophet
prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_model.fit(train_prophet)

# Forecast
future = prophet_model.make_future_dataframe(periods=len(test_prophet), freq='MS')
forecast_prophet = prophet_model.predict(future)

# Get test predictions
prophet_pred = forecast_prophet.iloc[train_size:train_size + len(test_prophet)]['yhat'].values
prophet_mae = mean_absolute_error(test, prophet_pred)

print(f"Prophet MAE: {prophet_mae:.2f}")

# Visualize
fig = prophet_model.plot(forecast_prophet)
plt.title('Prophet Forecast')
plt.show()

# Show components (trend, seasonality)
fig = prophet_model.plot_components(forecast_prophet)
plt.show()
```

---

## 3.6 Model Comparison Framework

### Systematic Comparison

```python
# Create comparison table
models_comparison = {
    'ARIMA(1,1,1)': arima_forecast,
    'SARIMA(1,1,1)(1,1,1,12)': sarima_forecast,
    'Prophet': prophet_pred
}

comparison_results = {}
for model_name, predictions in models_comparison.items():
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    comparison_results[model_name] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

results_df = pd.DataFrame(comparison_results).T.sort_values('MAE')
print("\nModel Comparison:")
print(results_df)

# Visualize all forecasts
plt.figure(figsize=(16, 7))
plt.plot(test.index, test, label='Actual', linewidth=2, marker='o', color='black')

for model_name, predictions in models_comparison.items():
    plt.plot(test.index, predictions, label=model_name, alpha=0.7)

plt.legend()
plt.title('Statistical Methods Comparison')
plt.grid(alpha=0.3)
plt.show()
```

### Choosing the Best Model

Consider:
1. **Accuracy** (MAE, RMSE, MAPE)
2. **Interpretability** (can you explain it?)
3. **Speed** (how fast does it train?)
4. **Confidence intervals** (uncertainty estimates)
5. **Robustness** (works on new data?)

---

## 3.7 Advanced Diagnostics

### Residual Analysis

```python
# Check residuals should look like white noise
residuals = test - fitted_model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Time series of residuals
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].grid()

# Histogram
axes[0, 1].hist(residuals, bins=20, edgecolor='black')
axes[0, 1].set_title('Residuals Distribution')

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 0])
axes[1, 0].set_title('ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Statistical tests
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box = acorr_ljungbox(residuals, lags=10)
print("Ljung-Box Test (should be > 0.05):")
print(ljung_box)
```

---

## 3.8 Mini Project: Build Statistical Forecasting Pipeline

### Project Steps

1. **Load and explore data** (visualize, check seasonality)
2. **Test stationarity** (ADF test)
3. **Apply differencing if needed** (make stationary)
4. **Read ACF/PACF** (identify parameters)
5. **Build ARIMA, SARIMA, Prophet** (3 models minimum)
6. **Use auto_arima** (automatic parameter selection)
7. **Compare on test set** (metrics table)
8. **Choose best model** (balance accuracy & interpretability)
9. **Analyze residuals** (check for patterns)
10. **Create forecast with intervals** (future predictions)

### Code Template

```python
# Complete pipeline
def statistical_forecasting_pipeline(data, test_size=0.2):
    """
    Full statistical forecasting pipeline
    """
    # Split
    split_idx = int(len(data) * (1 - test_size))
    train, test = data[:split_idx], data[split_idx:]
    
    # Test stationarity
    adf_result = adfuller(train)
    print(f"ADF p-value: {adf_result[1]:.4f}")
    
    # Auto ARIMA
    auto_model = auto_arima(train, stepwise=True, seasonal=False)
    arima_pred = auto_model.predict(n_periods=len(test))
    
    # SARIMA
    sarima_model = SARIMAX(train, order=auto_model.order, 
                          seasonal_order=(1,1,1,12)).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test))
    
    # Prophet
    df_p = pd.DataFrame({'ds': pd.date_range('2000', periods=len(data), freq='M'),
                        'y': data})
    pm = Prophet(yearly_seasonality=True, daily_seasonality=False)
    pm.fit(df_p[:split_idx])
    future = pm.make_future_dataframe(periods=len(test), freq='M')
    prophet_pred = pm.predict(future).iloc[split_idx:split_idx+len(test)]['yhat'].values
    
    # Evaluate
    metrics = {}
    for name, pred in [('ARIMA', arima_pred), ('SARIMA', sarima_pred), ('Prophet', prophet_pred)]:
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        metrics[name] = {'MAE': mae, 'RMSE': rmse}
    
    return metrics, (arima_pred, sarima_pred, prophet_pred)

# Run pipeline
metrics, predictions = statistical_forecasting_pipeline(df['Passengers'].values)
print(pd.DataFrame(metrics).T)
```

---

## 3.9 Key Takeaways

✅ **Do This:**
- Always test for stationarity first
- Use auto_arima for parameter selection
- Build multiple models (ARIMA, SARIMA, Prophet)
- Analyze residuals to verify model assumptions
- Include uncertainty intervals in forecasts

❌ **Don't Do This:**
- Assume d=0 (differencing = 0) without testing
- Ignore seasonal patterns
- Build only one model without comparison
- Interpret ACF/PACF incorrectly
- Forget to check if residuals are white noise

---

## Progress Checkpoint

**Completion: ~35%** ✓

You now understand:
- Why stationarity matters
- How to test and fix non-stationary data
- How to read ACF/PACF for parameter selection
- How to build ARIMA and SARIMA models
- When to use Prophet vs. traditional methods

**Next Module (Module 4):** Machine Learning for Forecasting
- Convert time series to supervised learning
- Feature engineering for ML
- Random Forest, XGBoost, ensemble methods

**Time Estimate for Module 3:** 8-10 hours  
**Estimated Combined Progress:** 35% → 50%

---

*End of Module 3: Statistical Time Series Methods*
