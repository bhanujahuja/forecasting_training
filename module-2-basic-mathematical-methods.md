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

## 2.4 Model Evaluation: Simple Metrics

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(df["Passengers"].dropna(), df["NaiveForecast"].dropna())
rmse = np.sqrt(mean_squared_error(df["Passengers"].dropna(), df["NaiveForecast"].dropna()))
print(f"Naive MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

---

## 2.5 Mini Project: Build and Compare Baseline Models

1. Use one of the provided datasets.
2. Implement mean, naive, and moving average forecasts.  
3. Plot results and compute metrics (MAE/RMSE).
4. Try exponential smoothing and regression.
5. Share your notebook & key findings in the course repo.

---

## 2.6 Summary

- Baseline methods help you diagnose time series patterns and provide a comparison point for advanced models.
- Understanding and testing these methods is critical before proceeding to statistical/ML techniques.

---

**Next Up:**  
Module 3: Statistical Time Series Methods (ARIMA, SARIMA, etc.)

---

*End of Module 2*