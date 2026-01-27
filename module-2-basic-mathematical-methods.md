# Module 2: Basic Mathematical Methods for Forecasting

This module introduces foundational mathematical approaches to forecasting, helping you build intuition and establish baseline models. These methods are simple, interpretable, and form the starting point for more advanced techniques.

---

## 2.1 Naive Forecasting Methods

### 2.1.1 Mean Forecast
- **Description:** Always predict the average value from the historical data.
- **Use Case:** Establishing a baseline for forecast accuracy.

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
mean_value = df["Passengers"].mean()
df["MeanForecast"] = mean_value
print(df[["Passengers", "MeanForecast"]].head())
```

---

### 2.1.2 Last Value (Naive) Forecast
- **Description:** Predict that the next value will be equal to the last observed value.
- **Use Case:** Works well for data with minimal trend/seasonality.

```python
df["NaiveForecast"] = df["Passengers"].shift(1)
print(df[["Passengers", "NaiveForecast"]].head())
```

---

### 2.1.3 Seasonal Naive Forecast
- **Description:** Predict that the next value will be equal to the value from the same season in the previous cycle (e.g., last year’s same month).
- **Use Case:** Strong seasonality, like retail sales or weather data.

```python
season_length = 12  # For monthly data with yearly seasonality
df["SeasonalNaiveForecast"] = df["Passengers"].shift(season_length)
print(df[["Passengers", "SeasonalNaiveForecast"]].head(season_length+5))
```

---

## 2.2 Smoothing Techniques

### 2.2.1 Moving Average Forecast
- **Description:** Predict using the average of the last N observations.
- **Use Case:** Smooths out short-term fluctuations, highlights longer-term trends.

```python
window = 3
df["MAForecast"] = df["Passengers"].rolling(window=window).mean().shift(1)
print(df[["Passengers", "MAForecast"]].head(window+5))
```

---

### 2.2.2 Exponential Smoothing

#### (a) Simple Exponential Smoothing
- **Description:** Weighs recent observations more heavily using a parameter α.
- **Use Case:** Data with no clear trend/seasonality.
- **Demo (with statsmodels):**
```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

train = df["Passengers"][:-12]
model = SimpleExpSmoothing(train).fit(smoothing_level=0.5, optimized=False)
forecast = model.forecast(12)
print(forecast)
```

#### (b) Double Exponential Smoothing (Holt’s Linear Trend)
- **Description:** Accounts for trends in the data.
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
forecast = model.forecast(12)
print(forecast)
```

#### (c) Triple Exponential Smoothing (Holt-Winters Method)
- **Description:** Accounts for trend and seasonality.
```python
model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
forecast = model.forecast(12)
print(forecast)
```

---

## 2.3 Regression-Based Forecasts

### 2.3.1 Linear Regression for Time-Based Forecasts
- **Description:** Use ordinary least squares regression with time as the predictor.
- **Demo:**
```python
from sklearn.linear_model import LinearRegression
import numpy as np

df = df.copy()
df["TimeIndex"] = np.arange(len(df))
X = df[["TimeIndex"]]
y = df["Passengers"]
model = LinearRegression().fit(X, y)
df["LinRegForecast"] = model.predict(X)
print(df[["Passengers", "LinRegForecast"]].head())
```

---

### 2.3.2 Multiple Linear Regression (Optional: for Multivariate Data)
- **Description:** If you have other variables (e.g., holidays, promotions), include those as features.
- **(Demo left as a challenge)**

---
---

## 2.5 Understanding Model Accuracy & Metrics

### 2.5.1 Why Multiple Metrics Matter
- **MAE (Mean Absolute Error):** Average absolute difference, easy to interpret, same units as data
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily, sensitive to outliers
- **MAPE (Mean Absolute Percentage Error):** Percentage-based, works well for comparing across scales
- **sMAPE (Symmetric MAPE):** Better symmetry properties than MAPE

```python
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np

def evaluate_forecast(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Example
metrics = evaluate_forecast(df["Passengers"].dropna(), df["NaiveForecast"].dropna())
print(metrics)
```

---

## 2.6 Train-Test Split for Time Series

### 2.6.1 Why Not Random Split?
- Time series has temporal order; randomization breaks dependencies
- Future data must remain in test set

### 2.6.2 Correct Approach
```python
# Split: Train on first 90%, test on last 10%
train_size = int(len(df) * 0.9)
train_data = df[:train_size]
test_data = df[train_size:]

# Fit on training data, evaluate on test
```

---

## 2.7 Mini Project: Build and Compare Baseline Models

### 2.7.1 Project Objectives
- Implement multiple baseline forecasting methods
- Compare their performance systematically
- Understand which method works best for your data

### 2.7.2 Complete Project Steps

1. **Choose your dataset**
   - Use airline passengers or your own dataset from Module 1
   - Ensure at least 50 observations for meaningful train-test split

2. **Implement baseline methods**
   - Mean forecast
   - Naive (last value) forecast
   - Seasonal naive (12-period lag for monthly data)
   - Moving average (window=3, 6, 12)
   - Exponential smoothing (Simple, Holt's, Holt-Winters)
   - Linear regression on time index

3. **Prepare train-test split**
   ```python
   train_size = int(len(df) * 0.8)
   train, test = df[:train_size], df[train_size:]
   ```

4. **Fit and forecast**
   - Fit each model on training data
   - Generate forecasts for test period
   - Store all forecasts in a results dataframe

5. **Evaluate and compare**
   - Calculate MAE, RMSE, MAPE for each method
   - Create a comparison table
   - Visualize actual vs. all forecasts

6. **Visualize results**
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(14, 6))
   plt.plot(test.index, test['Passengers'], label='Actual', linewidth=2)
   plt.plot(test.index, test['NaiveForecast'], label='Naive', alpha=0.7)
   plt.plot(test.index, test['MAForecast'], label='Moving Avg', alpha=0.7)
   # ... add more forecasts
   plt.legend()
   plt.title('Baseline Method Comparison')
   plt.xlabel('Time')
   plt.ylabel('Passengers')
   plt.show()
   ```

7. **Analysis and insights**
   - Which method performed best? Why?
   - How does model complexity affect accuracy?
   - Are there methods that work better for certain periods?
   - Which method is most interpretable and practical?

8. **Deliverables**
   - Complete Jupyter notebook with all steps
   - Comparison table (methods vs. error metrics)
   - Visualization plots
   - Summary of findings (150-200 words)

---

## 2.8 Summary

- Baseline methods help you diagnose time series patterns and provide a comparison point for advanced models.
- Understanding and testing these methods is critical before proceeding to statistical/ML techniques.
- Always use proper train-test splits respecting time order.
- Evaluate using multiple metrics and visualize results.

---

**Next Up:**  
Module 3: Statistical Time Series Methods (ARIMA, SARIMA, etc.)

---

*End of Module 2*