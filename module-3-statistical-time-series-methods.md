# Module 3: Statistical Time Series Methods

This module covers core statistical time series forecasting methods. These models explain and predict patterns in sequential data, handling trend, seasonality, and inter-variable relationships.

---

## 3.1 Introduction to Statistical Time Series Methods

- **Key concepts:** Stationarity, trend, seasonality, autocorrelation, lag, differencing
- **Why use statistical models?**  
  - Transparent, interpretable  
  - Efficient for smaller data and structured time dependence  
  - Baseline for ML/AI comparisons

---

## 3.2 ARIMA Models

### 3.2.1 What is ARIMA?
- **ARIMA = AutoRegressive Integrated Moving Average**
- Components:
    - **AR (p):** Autoregression — using past values
    - **I (d):** Integration — differencing for stationarity
    - **MA (q):** Moving Average — past error terms
- **Notation:** ARIMA(p, d, q)
- **Use case:** Univariate, regular time series without strong seasonality

---

### 3.2.2 Steps for ARIMA Modeling

1. **Make your series stationary:**  
   - Visual inspection  
   - Augmented Dickey-Fuller (ADF) test (`statsmodels.tsa.stattools.adfuller`)
   - Differencing (d>0 if non-stationary)
2. **Identify p and q:**  
   - Look at Autocorrelation (ACF) and Partial ACF (PACF) plots
3. **Fit ARIMA:**  
   - Use `statsmodels.tsa.arima.model.ARIMA`
4. **Forecast and evaluate**

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
series = df["Passengers"]

# Stationarity test
result = adfuller(series)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Differencing (if needed)
diff_series = series.diff().dropna()

# ACF and PACF
plot_acf(diff_series)
plot_pacf(diff_series)
plt.show()

# Fit ARIMA (example: ARIMA(1,1,1))
model = ARIMA(series, order=(1,1,1)).fit()
pred = model.forecast(steps=12)
print(pred)
```

---

### 3.2.3 Diagnostics & Hyperparameter Tuning

- **Check residuals:** Plot, should resemble white noise
- **AIC/BIC:** Use for model selection
- **Grid search over (p,d,q) with auto_arima (`pmdarima`)**

---

## 3.3 SARIMA: Seasonal ARIMA

- **SARIMA = Seasonal ARIMA = ARIMA + seasonality**
- **Notation:** SARIMA(p, d, q)(P, D, Q, s)
    - s = length of season (e.g., 12 for months in year)
- Use when data shows repeated seasonal patterns

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_forecast = sarima_model.forecast(steps=12)
print(sarima_forecast)
```

---

## 3.4 Other Classical Models

### 3.4.1 Vector AutoRegression (VAR)
- **Use case:** Multivariate time series (multiple variables, like temperature and sales)
```python
from statsmodels.tsa.api import VAR

df_multi = df[["Passengers"]] # replace with multiple columns for real use
model = VAR(df_multi)
model_fitted = model.fit(maxlags=15, ic='aic')
lag_order = model_fitted.k_ar
forecast_input = df_multi.values[-lag_order:]
forecast = model_fitted.forecast(y=forecast_input, steps=12)
```

### 3.4.2 State Space Models & Kalman Filter (Brief)
- Used for complex, real-time, or partially observed systems

---

### 3.4.3 Prophet (by Meta/Facebook)
- Decomposes into trend + seasonality + holidays
- Handles missing data, outliers, and change points easily

```python
from prophet import Prophet
df_prophet = df.rename(columns={"Month": "ds", "Passengers": "y"})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)
model.plot(forecast)
```

---

---

## 3.5 Model Evaluation & Diagnostics

- **Residual analysis** (should look random)
- **Forecast error metrics:** MAE, RMSE, MAPE
- **Visual comparison:** Plot actual vs. predicted

### 3.5.1 Residual Diagnostics
```python
# Check residuals
residuals = arima_model.resid
residuals.plot(title='ARIMA Residuals')
residuals.plot(kind='kde', title='Residual Distribution')
plt.show()

# Statistical tests
from statsmodels.stats.diagnostic import acorr_ljungbox
print(acorr_ljungbox(residuals, lags=10))  # Should be > 0.05 (white noise)
```

---

## 3.6 Comparing ARIMA, SARIMA, and Prophet

### 3.6.1 When to Use Each Method
- **ARIMA:** Simple univariate series, minimal seasonality
- **SARIMA:** Clear seasonal patterns, univariate data
- **Prophet:** Trend breaks, holidays, missing data, multivariate features

### 3.6.2 Practical Comparison Framework
```python
# Fit multiple models
models = {
    'ARIMA(1,1,1)': ARIMA(series, order=(1,1,1)).fit(),
    'SARIMA(1,1,1)(1,1,1,12)': SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(),
    'Prophet': Prophet().fit(df_prophet)
}

# Compare forecasts
results = pd.DataFrame()
for name, model in models.items():
    if name == 'Prophet':
        future = model.make_future_dataframe(periods=12, freq='ME')
        forecast = model.predict(future)['yhat'].tail(12).values
    else:
        forecast = model.forecast(steps=12)
    results[name] = forecast
```

---

## 3.7 Hyperparameter Tuning with Grid Search

### 3.7.1 Auto ARIMA
```python
from pmdarima import auto_arima

# Automatically select best (p,d,q)
stepwise_model = auto_arima(series, start_p=0, start_q=0, max_p=5, max_q=5,
                            stepwise=True, seasonal=False, trace=True)
print(stepwise_model.summary())
```

### 3.7.2 Manual Grid Search
```python
# Grid search for best ARIMA parameters
best_aic = float('inf')
best_params = None

for p in range(0, 4):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                model = ARIMA(series, order=(p,d,q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_params = (p, d, q)
            except:
                pass

print(f"Best ARIMA{best_params} with AIC {best_aic:.2f}")
```

---

## 3.8 Mini-Project: End-to-End Time Series Forecasting

### 3.8.1 Project Objectives
- Build a complete forecasting pipeline
- Compare multiple statistical models
- Deploy forecasts with uncertainty intervals
- Create actionable insights

### 3.8.2 Complete Project Steps

1. **Data Selection & Preparation**
   - Choose a time series dataset (airline passengers, energy, stock price, etc.)
   - Load and inspect data
   - Check for missing values, outliers
   - Perform basic EDA (visualizations, summary stats)

2. **Stationarity Testing**
   ```python
   from statsmodels.tsa.stattools import adfuller, kpss
   
   # Test with both ADF and KPSS
   adf = adfuller(series)
   kpss_result = kpss(series, regression='c')
   
   print(f"ADF p-value: {adf[1]:.4f}, KPSS p-value: {kpss_result[1]:.4f}")
   ```

3. **Series Decomposition**
   ```python
   from statsmodels.tsa.seasonal import seasonal_decompose
   
   decomposition = seasonal_decompose(series, model='additive', period=12)
   decomposition.plot()
   plt.show()
   ```

4. **ACF/PACF Analysis**
   - Create ACF and PACF plots
   - Identify potential p, d, q values
   - Document observations

5. **Model Building (Fit 3 models)**
   
   **Model 1: ARIMA with auto_arima**
   ```python
   from pmdarima import auto_arima
   arima_model = auto_arima(series, seasonal=False, stepwise=True)
   arima_forecast = arima_model.forecast(steps=12, return_conf_int=True)
   ```
   
   **Model 2: SARIMA (if seasonal)**
   ```python
   sarima_model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
   sarima_forecast = sarima_model.get_forecast(steps=12)
   ```
   
   **Model 3: Prophet**
   ```python
   prophet_model = Prophet()
   prophet_model.fit(df_prophet)
   future = prophet_model.make_future_dataframe(periods=12, freq='ME')
   prophet_forecast = prophet_model.predict(future)
   ```

6. **Model Evaluation (on test set)**
   - Split data: 80% train, 20% test
   - Fit models on training data
   - Generate test-set forecasts
   - Calculate MAE, RMSE, MAPE
   - Create comparison table

7. **Visualization & Results**
   ```python
   # Plot all forecasts vs. actual
   plt.figure(figsize=(15, 6))
   plt.plot(test.index, test['value'], label='Actual', linewidth=2)
   plt.plot(forecast_index, arima_fc, label='ARIMA')
   plt.plot(forecast_index, sarima_fc, label='SARIMA')
   plt.plot(forecast_index, prophet_fc, label='Prophet')
   
   # Add confidence intervals
   plt.fill_between(forecast_index, lower, upper, alpha=0.2)
   plt.legend()
   plt.title('Statistical Time Series Forecast Comparison')
   plt.show()
   ```

8. **Residual Diagnostics**
   - Plot residuals over time
   - Check for autocorrelation (Ljung-Box test)
   - Verify normality
   - Document any patterns

9. **Generate Future Forecast**
   - Use best-performing model
   - Forecast 12 periods ahead (or relevant horizon)
   - Include confidence intervals
   - Document assumptions

10. **Interpretation & Insights**
    - Which model performed best and why?
    - What patterns does the forecast show?
    - What assumptions underlie the forecast?
    - What are limitations and risks?

### 3.8.3 Deliverables

- **Jupyter notebook** with all code, comments, visualizations
- **Model comparison table** (methods, metrics, AIC/BIC)
- **Forecast visualization** with confidence intervals
- **Residual diagnostics plots**
- **Written summary** (300-400 words) covering:
  - Data description
  - Model selection rationale
  - Key findings
  - Forecast interpretation
  - Limitations and next steps

### 3.8.4 Expected Learning Outcomes

- Understand stationarity and differencing
- Apply ACF/PACF for model selection
- Implement ARIMA, SARIMA, and Prophet
- Compare models systematically
- Interpret forecasts with uncertainty
- Diagnose model fit through residuals

---

## 3.9 Advanced Topics (Optional Deep Dives)

### 3.9.1 Multiple Forecasting Horizons
- 1-step ahead (next period)
- Multi-step ahead (rolling window)
- Direct forecasting vs. recursive

### 3.9.2 Ensemble Methods
- Combine ARIMA + SARIMA + Prophet
- Use weighted averaging based on test performance
- Reduce variance through ensemble

### 3.9.3 Handling Structural Breaks
```python
# Prophet handles automatically
# Or detect manually with change point analysis
from statsmodels.stats.diagnostic import BreakpointUnitRoot
```

---

## 3.10 Next Up

Machine Learning for forecasting! Advanced feature engineering and non-linear models in Module 4.

---

## 3.7 Summary

- Statistical models like ARIMA and SARIMA are useful, interpretable, and often the first choice.
- Always start with good diagnostics and baseline comparisons.
- Modular tools like Prophet simplify modeling with trend/seasonality/holiday effects.

---

**Next Up:**  
Module 4: Machine Learning Methods for Forecasting

---

*End of Module 3*