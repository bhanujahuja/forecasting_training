# Module 2: Basic Mathematical Methods for Forecasting

**Duration:** 6-8 hours (including mini-project)  
**Prerequisites:** Modules 0-1  
**Learning Level:** Beginner to Intermediate

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand** why baseline methods are essential starting points
2. **Implement** naive, smoothing, and regression-based forecasting methods
3. **Choose** the appropriate baseline method for different data patterns
4. **Evaluate** forecast accuracy using multiple metrics
5. **Recognize** the importance of proper time-series train-test splitting
6. **Build** and compare multiple baseline models systematically
7. **Interpret** results and communicate findings clearly

---

## Why This Module Matters

### The Baseline First Principle

Before building complex models (ARIMA, Machine Learning, Deep Learning), you must:

- **Establish performance floor:** What's the minimum accuracy to beat?
- **Identify data patterns:** What does a simple method capture?
- **Build intuition:** Understand your data's behavior
- **Avoid overfitting:** More complex ≠ better
- **Save computational resources:** Simple methods are fast

### Real-World Impact

```
Scenario: You build a complex LSTM model with 89% accuracy
But the naive forecast (repeat last value) has 87% accuracy

Did your complex model justify the effort?
  - More computational cost
  - Harder to interpret
  - Slower to retrain
  - Requires more data

Answer: Maybe not! Always start simple.
```

---

## Quick Reference: When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Mean** | No trend/seasonality | Very simple | Ignores all patterns |
| **Naive** | Random walk behavior | Captures persistence | No learned pattern |
| **Seasonal Naive** | Strong seasonality | Captures repeating cycles | Misses trends |
| **Moving Avg** | Noisy data, trending | Smooth, stable | Lags behind changes |
| **Exponential Smoothing** | Trend + seasonality | Flexible, adaptive | More parameters |
| **Linear Regression** | Consistent trends | Interpretable, fast | Assumes linearity |

---

## 2.1 Naive Forecasting Methods

### 2.1.1 Mean Forecast

**What it does:** Always predict the average value from the historical data.

**When to use:**
- Establishing a baseline for forecast accuracy
- When data has no clear trend or seasonality
- Quick sanity check: "How bad is predicting the mean?"

**How it works:**
```
Historical data: [10, 12, 11, 13, 12]
Mean = 11.6

Forecast for next 5 periods: [11.6, 11.6, 11.6, 11.6, 11.6]
```

**Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df['Month'] = pd.to_datetime(df['Month'])

# Calculate mean
mean_value = df["Passengers"].mean()
print(f"Mean of historical data: {mean_value:.2f}")

# Create forecast
df["MeanForecast"] = mean_value

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['Month'], df['Passengers'], label='Actual', linewidth=2)
plt.axhline(y=mean_value, color='r', linestyle='--', label='Mean Forecast')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.title('Mean Forecast')
plt.show()

# Evaluation on test set
test_start = int(len(df) * 0.8)
test_actual = df['Passengers'][test_start:]
test_pred = pd.Series([mean_value] * len(test_actual), index=test_actual.index)

mae = mean_absolute_error(test_actual, test_pred)
rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

**Key Insight:**
The mean forecast is almost never the best choice, but it provides a critical benchmark. If your sophisticated model doesn't beat this, something is wrong.

---

### 2.1.2 Last Value (Naive) Forecast

**What it does:** Predict that the next value will be equal to the last observed value.

**When to use:**
- Data follows a "random walk" (each point builds on previous)
- Stock prices, exchange rates, random walk processes
- Quick baseline when data shows strong persistence
- "What if we just stay put?"

**How it works:**
```
Historical data: [10, 15, 12, 18, 20]
Last value: 20

Forecast for next 5 periods: [20, 20, 20, 20, 20]
(Or more realistically: next point = 20)
```

**Mathematical Intuition:**

A random walk means: **y(t+1) = y(t) + ε** where ε is random noise

If the expected noise is 0, the best guess for y(t+1) is just y(t).

**Implementation:**

```python
# Simple approach
df["NaiveForecast"] = df["Passengers"].shift(1)

# More controlled version
def naive_forecast(series, test_start_idx):
    """
    Generate naive forecasts for test period
    """
    forecasts = []
    for i in range(test_start_idx, len(series)):
        # Forecast = last observed value
        forecast = series[i-1]
        forecasts.append(forecast)
    return np.array(forecasts)

test_idx = int(len(df) * 0.8)
test_actual = df['Passengers'][test_idx:].values
test_pred = naive_forecast(df['Passengers'].values, test_idx)

# Evaluate
from sklearn.metrics import mean_absolute_percentage_error
mae = mean_absolute_error(test_actual, test_pred)
mape = mean_absolute_percentage_error(test_actual, test_pred)
print(f"MAE: {mae:.2f}, MAPE: {mape:.2f}%")

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o')
plt.plot(df['Month'][test_idx:], test_pred, label='Naive Forecast', marker='s', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend()
plt.title('Naive (Last Value) Forecast')
plt.show()
```

**Real-World Example:**

Stock market: If yesterday's price was $100, the naive forecast for today is $100.
- Simple? Yes
- Often surprisingly accurate? Yes, for short-term predictions
- Why? Because markets are (partially) efficient and unpredictable

---

### 2.1.3 Seasonal Naive Forecast

**What it does:** Predict that the next value will equal the value from the same season in the previous cycle.

**When to use:**
- Strong seasonal patterns (same month last year)
- Retail sales, weather, energy demand
- When you need to capture "what happened last year, this time"

**How it works:**
```
Example: Monthly data with yearly seasonality (period = 12)
Jan, Feb, Mar, ..., Dec | Jan, Feb, Mar, ..., Dec | ...

Forecast for Jan (month 25) = Jan from year 1 (month 1)
Forecast for Feb (month 26) = Feb from year 1 (month 2)
```

**Why this works:**
Seasonality = repeating patterns. If sales spike every July, predict July sales = last July's sales.

**Implementation:**

```python
# Simple version
season_length = 12  # Monthly data with yearly seasonality
df["SeasonalNaiveForecast"] = df["Passengers"].shift(season_length)

# Visualize the seasonal pattern
def seasonal_naive_forecast(series, season_length, test_start_idx):
    """
    Generate seasonal naive forecasts
    """
    forecasts = []
    for i in range(test_start_idx, len(series)):
        # Forecast = value from same season last year
        if i >= season_length:
            forecast = series[i - season_length]
        else:
            forecast = np.nan  # Can't forecast without historical season
        forecasts.append(forecast)
    return np.array(forecasts)

test_idx = int(len(df) * 0.8)
test_actual = df['Passengers'][test_idx:].values
test_pred = seasonal_naive_forecast(df['Passengers'].values, 12, test_idx)

# Remove NaN values
valid_idx = ~np.isnan(test_pred)
test_actual_valid = test_actual[valid_idx]
test_pred_valid = test_pred[valid_idx]

mae = mean_absolute_error(test_actual_valid, test_pred_valid)
mape = mean_absolute_percentage_error(test_actual_valid, test_pred_valid)
print(f"MAE: {mae:.2f}, MAPE: {mape:.4f}")

# Visualization showing seasonal pattern
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Full view
axes[0].plot(df['Month'], df['Passengers'], label='Actual', linewidth=2)
axes[0].plot(df['Month'][test_idx:], test_pred, label='Seasonal Naive', alpha=0.7)
axes[0].set_title('Seasonal Naive Forecast - Full View')
axes[0].legend()
axes[0].grid()

# Plot 2: Zoom in on test set
axes[1].plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o')
axes[1].plot(df['Month'][test_idx:], test_pred, label='Forecast', marker='s', alpha=0.7)
axes[1].set_title('Seasonal Naive Forecast - Test Period')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()
```

**How to Determine Season Length:**
```python
# Visual inspection
df['Passengers'].plot(figsize=(14, 6))
plt.title('Look for repeating patterns')
plt.show()

# For monthly data:
#   - 12 = yearly seasonality (Jan repeats every 12 months)
#   - 4 = quarterly seasonality
#   - 7 = weekly seasonality (for daily data)

# Autocorrelation analysis
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Passengers'], lags=40)
plt.title('ACF - peaks at lag = season length')
plt.show()
```

**Pitfall to Avoid:**
Seasonal naive doesn't capture trends! If sales grow over time, seasonal naive won't show growth.

---

## 2.2 Smoothing Techniques

### 2.2.1 Moving Average

**What it does:** Predict using the average of the last N observations.

**Mathematical Formula:**
```
MA(t) = (y(t) + y(t-1) + ... + y(t-N+1)) / N
```

**When to use:**
- Remove noise from data
- Identify underlying trends
- When recent observations matter equally
- Smooth volatile data

**Implementation:**

```python
def moving_average_forecast(series, window_size, test_start_idx):
    """
    Generate moving average forecasts
    """
    forecasts = []
    for i in range(test_start_idx, len(series)):
        # Average of last N values
        ma = series[max(0, i-window_size):i].mean()
        forecasts.append(ma)
    return np.array(forecasts)

# Test multiple window sizes
windows = [3, 6, 12]
fig, axes = plt.subplots(len(windows), 1, figsize=(14, 10))

test_idx = int(len(df) * 0.8)
test_actual = df['Passengers'][test_idx:].values

for idx, window in enumerate(windows):
    test_pred = moving_average_forecast(df['Passengers'].values, window, test_idx)
    mae = mean_absolute_error(test_actual, test_pred)
    
    axes[idx].plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o')
    axes[idx].plot(df['Month'][test_idx:], test_pred, label=f'MA(window={window})', alpha=0.7)
    axes[idx].set_title(f'Moving Average (window={window}) - MAE={mae:.2f}')
    axes[idx].legend()
    axes[idx].grid()

plt.tight_layout()
plt.show()

# Find best window
best_mae = float('inf')
best_window = None
for window in range(1, 24):
    pred = moving_average_forecast(df['Passengers'].values, window, test_idx)
    mae = mean_absolute_error(test_actual, pred)
    if mae < best_mae:
        best_mae = mae
        best_window = window

print(f"Best window size: {best_window} with MAE: {best_mae:.2f}")
```

**Window Size Selection:**
- Small window (3-6): Follows data closely, includes noise
- Large window (12+): Smooth, but lags behind changes
- Rule of thumb: Start with 2-3 seasonal periods

---

### 2.2.2 Exponential Smoothing

**What it does:** Weights recent observations more heavily using a parameter α (alpha).

**Formula:**
```
S(t) = α·y(t) + (1-α)·S(t-1)

Where:
- α (0 to 1): Smoothing parameter
  - α close to 1: Recent data weighted heavily
  - α close to 0: Past data weighted heavily
```

**When to use:**
- Adaptive to changes (more recent = more important)
- Simple univariate data
- When you need weights that decay over time

#### Simple Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def exponential_smoothing_forecast(series, alpha, test_start_idx):
    """
    Simple exponential smoothing
    """
    # Fit on training data
    train = series[:test_start_idx]
    model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
    
    # Forecast for test period
    forecasts = []
    current_level = train.iloc[-1]
    
    for i in range(test_start_idx, len(series)):
        forecast = current_level
        forecasts.append(forecast)
        # Update level with new observation
        current_level = alpha * series[i] + (1 - alpha) * current_level
    
    return np.array(forecasts)

# Test different alpha values
alphas = [0.2, 0.5, 0.9]
fig, axes = plt.subplots(len(alphas), 1, figsize=(14, 10))

test_idx = int(len(df) * 0.8)
train = df['Passengers'][:test_idx]
test_actual = df['Passengers'][test_idx:].values

for idx, alpha in enumerate(alphas):
    # Use statsmodels for cleaner implementation
    model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
    test_pred = model.forecast(steps=len(test_actual))
    mae = mean_absolute_error(test_actual, test_pred)
    
    axes[idx].plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o')
    axes[idx].plot(df['Month'][test_idx:], test_pred.values, label=f'SES (α={alpha})', alpha=0.7)
    axes[idx].set_title(f'Simple Exponential Smoothing (α={alpha}) - MAE={mae:.2f}')
    axes[idx].legend()
    axes[idx].grid()

plt.tight_layout()
plt.show()

# Auto-optimize alpha
model_optimized = SimpleExpSmoothing(train).fit(optimized=True)
print(f"Optimized alpha: {model_optimized.params['smoothing_level']:.4f}")
```

---

#### Double Exponential Smoothing (Holt's Linear Trend)

**What it does:** Adds trend component to simple exponential smoothing.

**When to use:**
- Data with consistent upward/downward trend
- No seasonality
- Need to capture trend direction and magnitude

**Implementation:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Double Exponential Smoothing (Holt's method)
train = df['Passengers'][:test_idx]
model_holt = ExponentialSmoothing(
    train,
    trend='add',  # Additive trend
    seasonal=None,  # No seasonality
    damped_trend=False
).fit()

test_pred_holt = model_holt.forecast(steps=len(test_actual))
mae_holt = mean_absolute_error(test_actual, test_pred_holt)

plt.figure(figsize=(14, 6))
plt.plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o')
plt.plot(df['Month'][test_idx:], test_pred_holt.values, label='Holt (trend)', alpha=0.7)
plt.title(f'Double Exponential Smoothing (Holt) - MAE={mae_holt:.2f}')
plt.legend()
plt.grid()
plt.show()
```

---

#### Triple Exponential Smoothing (Holt-Winters)

**What it does:** Adds both trend AND seasonality components.

**When to use:**
- Data with trend + seasonal patterns (most real-world data)
- Clear repeating cycles plus overall direction

**Implementation:**

```python
# Triple Exponential Smoothing (Holt-Winters)
model_hw = ExponentialSmoothing(
    train,
    trend='add',  # Additive trend
    seasonal='add',  # Additive seasonality
    seasonal_periods=12  # Yearly seasonality for monthly data
).fit()

test_pred_hw = model_hw.forecast(steps=len(test_actual))
mae_hw = mean_absolute_error(test_actual, test_pred_hw)

# Compare all methods
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['Month'][test_idx:], test_actual, label='Actual', marker='o', linewidth=2)
ax.plot(df['Month'][test_idx:], test_pred_holt.values, label='Holt (trend only)', alpha=0.7)
ax.plot(df['Month'][test_idx:], test_pred_hw.values, label='Holt-Winters (trend+seasonal)', alpha=0.7)
ax.set_title('Comparing Exponential Smoothing Methods')
ax.legend()
ax.grid()
plt.show()

print(f"Holt MAE: {mae_holt:.2f}")
print(f"Holt-Winters MAE: {mae_hw:.2f}")
```

---

## 2.3 Regression-Based Forecasts

### 2.3.1 Linear Regression for Trend Forecasting

**What it does:** Fit a line to data using time index as predictor.

**Mathematical Formula:**
```
y(t) = β₀ + β₁·t + ε

Where:
- t = time index (1, 2, 3, ...)
- β₀ = intercept
- β₁ = slope (trend direction)
- ε = error term
```

**When to use:**
- Clear linear trend over time
- Simple, interpretable trend baseline
- Quick trend quantification

**Implementation:**

```python
from sklearn.linear_model import LinearRegression

# Prepare data
train_idx = int(len(df) * 0.8)
train_data = df[:train_idx].copy()
test_data = df[train_idx:].copy()

# Create time index
train_data['time_idx'] = np.arange(len(train_data))
test_data['time_idx'] = np.arange(len(train_data), len(df))

# Fit linear regression
X_train = train_data[['time_idx']].values
y_train = train_data['Passengers'].values

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# Forecast
X_test = test_data[['time_idx']].values
test_pred_linear = model_linear.predict(X_test)
mae_linear = mean_absolute_error(test_data['Passengers'].values, test_pred_linear)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1: Full data with trend line
all_time = np.arange(len(df)).reshape(-1, 1)
all_pred = model_linear.predict(all_time)

axes[0].plot(df['Month'], df['Passengers'], label='Actual', linewidth=2)
axes[0].plot(df['Month'], all_pred, label='Linear Trend', linestyle='--', color='red')
axes[0].set_title(f'Linear Regression Trend: y = {model_linear.intercept_:.2f} + {model_linear.coef_[0]:.4f}·t')
axes[0].legend()
axes[0].grid()

# Plot 2: Test period comparison
axes[1].plot(df['Month'][train_idx:], test_data['Passengers'].values, label='Actual', marker='o')
axes[1].plot(df['Month'][train_idx:], test_pred_linear, label='Forecast', marker='s', alpha=0.7)
axes[1].set_title(f'Linear Regression Forecast - MAE={mae_linear:.2f}')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# Interpretation
print(f"Intercept: {model_linear.intercept_:.2f}")
print(f"Slope: {model_linear.coef_[0]:.4f} (passengers per month)")
print(f"Interpretation: On average, {model_linear.coef_[0]:.2f} more passengers per month")
```

**Important Limitations:**
- Assumes perfectly linear trend (unrealistic)
- Doesn't capture seasonality
- Ignores non-linear patterns
- Use only when trend is truly linear

---

## 2.4 Understanding Evaluation Metrics

### 2.4.1 Why Multiple Metrics?

Different metrics capture different aspects:

| Metric | Formula | Interpretation | Best For |
|--------|---------|-----------------|----------|
| **MAE** | Σ\|y - ŷ\| / n | Avg. error in original units | Business-friendly |
| **RMSE** | √(Σ(y - ŷ)² / n) | Penalizes large errors | When large errors are costly |
| **MAPE** | Σ(\|y - ŷ\| / y) / n | Percentage error | Comparing across scales |
| **MASE** | MAE / naive MAE | Relative to naive baseline | Comparing to benchmark |

**Example:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def evaluate_all_metrics(actual, predicted):
    """
    Calculate all important metrics
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    
    # MASE: Mean Absolute Scaled Error (relative to naive)
    naive_pred = np.array([actual[i-1] if i > 0 else actual[0] for i in range(len(actual))])
    naive_mae = mean_absolute_error(actual, naive_pred)
    mase = mae / naive_mae if naive_mae > 0 else np.inf
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase
    }

# Compare different methods
methods = {
    'Naive': naive_pred,
    'Seasonal Naive': seasonal_naive_pred,
    'Moving Average': ma_pred,
    'Linear Regression': linear_pred
}

results = {}
for method_name, predictions in methods.items():
    results[method_name] = evaluate_all_metrics(test_actual, predictions)

results_df = pd.DataFrame(results).T
print(results_df)
print("\nInterpretation:")
print("- Lowest MAE/RMSE/MAPE = best accuracy")
print("- MASE < 1.0 = better than naive baseline")
```

---

## 2.5 Proper Train-Test Splitting for Time Series

### 2.5.1 Why Not Random Shuffle?

❌ **WRONG:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# This shuffles data, breaking temporal order!
```

**Why it's wrong:**
- Breaks time dependency
- Allows "information leakage" from future to past
- Creates unrealistic validation

✅ **CORRECT:**
```python
# Sequential split: train on past, test on future
split_idx = int(len(df) * 0.8)
train = df[:split_idx]
test = df[split_idx:]

X_train, y_train = train[features], train['target']
X_test, y_test = test[features], test['target']
```

### 2.5.2 Time Series Cross-Validation

For more robust evaluation, use expanding window or sliding window cross-validation:

```python
from sklearn.model_selection import TimeSeriesSplit

# Expanding window: train set grows
tscv = TimeSeriesSplit(n_splits=5)

fold_results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate
    model = YourForecastingModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    fold_results.append({
        'fold': fold + 1,
        'train_size': len(train_idx),
        'test_size': len(test_idx),
        'MAE': mae
    })

cv_results = pd.DataFrame(fold_results)
print(cv_results)
print(f"Average MAE: {cv_results['MAE'].mean():.2f}")
```

**Visualizing the splits:**
```python
fig, axes = plt.subplots(tscv.n_splits, 1, figsize=(14, 8))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    ax = axes[fold]
    
    # Plot training and test periods
    ax.bar(train_idx, [1] * len(train_idx), color='blue', alpha=0.5, label='Train')
    ax.bar(test_idx, [1] * len(test_idx), color='red', alpha=0.5, label='Test')
    
    ax.set_xlim(-1, len(X))
    ax.set_ylim(0, 1.2)
    ax.set_title(f'Fold {fold + 1}')
    ax.legend(loc='upper right')
    ax.set_yticks([])

plt.tight_layout()
plt.show()
```

---

## 2.6 Mini Project: Build and Compare Baseline Models

### 2.6.1 Project Objectives

- Implement multiple baseline methods
- Compare performance systematically
- Understand which method works best for YOUR data
- Gain intuition about data patterns

### 2.6.2 Step-by-Step Walkthrough

#### Step 1: Load and Explore Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Month'].min()} to {df['Month'].max()}")
print(f"\nBasic Statistics:\n{df['Passengers'].describe()}")

# Visualize full series
plt.figure(figsize=(14, 6))
plt.plot(df['Month'], df['Passengers'], marker='o', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.title('Full Time Series')
plt.grid()
plt.show()

# Check for seasonality pattern
fig, ax = plt.subplots(figsize=(14, 6))
for year in range(1949, 1961):
    year_data = df[df['Month'].dt.year == year]['Passengers'].values
    ax.plot(range(1, 13), year_data, marker='o', label=str(year), alpha=0.7)
ax.set_xlabel('Month')
ax.set_ylabel('Passengers')
ax.set_title('Seasonal Pattern by Year')
ax.legend(ncol=4)
ax.grid()
plt.show()
```

#### Step 2: Prepare Train-Test Split

```python
# 80-20 split
train_size = int(len(df) * 0.8)
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

train_series = train_df['Passengers'].values
test_series = test_df['Passengers'].values

print(f"Train size: {len(train_series)} samples")
print(f"Test size: {len(test_series)} samples")
```

#### Step 3: Implement All Baseline Methods

```python
# Dictionary to store all predictions
all_forecasts = {}

# 1. Mean Forecast
mean_value = train_series.mean()
all_forecasts['Mean'] = np.full(len(test_series), mean_value)

# 2. Naive (Last Value)
all_forecasts['Naive'] = np.full(len(test_series), train_series[-1])

# 3. Seasonal Naive (12-month lag)
season_length = 12
seasonal_naive_preds = []
for i in range(len(test_series)):
    if train_size - season_length + i >= 0:
        seasonal_naive_preds.append(df['Passengers'].iloc[train_size - season_length + i])
    else:
        seasonal_naive_preds.append(mean_value)  # Fallback
all_forecasts['Seasonal Naive'] = np.array(seasonal_naive_preds)

# 4. Moving Average (window=3, 6, 12)
for window in [3, 6, 12]:
    ma_preds = []
    for i in range(len(test_series)):
        if train_size - window + i >= 0:
            ma = df['Passengers'].iloc[max(0, train_size - window + i):train_size + i].mean()
            ma_preds.append(ma)
        else:
            ma_preds.append(train_series[-window:].mean())
    all_forecasts[f'MA({window})'] = np.array(ma_preds)

# 5. Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

ses_model = SimpleExpSmoothing(train_series).fit(optimized=True)
all_forecasts['SES'] = ses_model.forecast(steps=len(test_series)).values

hw_model = ExponentialSmoothing(
    train_series,
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()
all_forecasts['Holt-Winters'] = hw_model.forecast(steps=len(test_series)).values

# 6. Linear Regression
from sklearn.linear_model import LinearRegression

X_train = np.arange(len(train_series)).reshape(-1, 1)
X_test = np.arange(len(train_series), len(train_series) + len(test_series)).reshape(-1, 1)

lr_model = LinearRegression()
lr_model.fit(X_train, train_series)
all_forecasts['Linear Regression'] = lr_model.predict(X_test)
```

#### Step 4: Evaluate All Methods

```python
def calculate_all_metrics(actual, predicted):
    """Calculate MAE, RMSE, MAPE"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Evaluate all methods
evaluation_results = {}
for method_name, predictions in all_forecasts.items():
    evaluation_results[method_name] = calculate_all_metrics(test_series, predictions)

# Create results table
results_table = pd.DataFrame(evaluation_results).T.sort_values('MAE')
results_table['MAE'] = results_table['MAE'].round(2)
results_table['RMSE'] = results_table['RMSE'].round(2)
results_table['MAPE'] = results_table['MAPE'].round(4)

print("\n" + "="*60)
print("BASELINE METHODS COMPARISON")
print("="*60)
print(results_table)
print("="*60)
```

#### Step 5: Visualize All Forecasts

```python
# Plot all methods
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual values
ax.plot(test_df['Month'], test_series, label='Actual', marker='o', linewidth=3, color='black')

# Plot all forecasts
colors = plt.cm.tab20(np.linspace(0, 1, len(all_forecasts)))
for (method_name, predictions), color in zip(all_forecasts.items(), colors):
    ax.plot(test_df['Month'], predictions, label=method_name, alpha=0.7, color=color)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Passengers', fontsize=12)
ax.set_title('Comparison of All Baseline Methods', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Separate visualization by complexity
simple_methods = ['Mean', 'Naive', 'Seasonal Naive']
smoothing_methods = ['MA(3)', 'MA(6)', 'MA(12)', 'SES', 'Holt-Winters']
regression_methods = ['Linear Regression']

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

for ax, methods, title in zip(axes, [simple_methods, smoothing_methods, regression_methods],
                               ['Simple Methods', 'Smoothing Methods', 'Regression Methods']):
    ax.plot(test_df['Month'], test_series, label='Actual', marker='o', linewidth=2, color='black')
    
    for method in methods:
        if method in all_forecasts:
            ax.plot(test_df['Month'], all_forecasts[method], label=method, alpha=0.7)
    
    mae_values = {m: evaluation_results[m]['MAE'] for m in methods if m in evaluation_results}
    ax.set_title(f'{title} - Best: {min(mae_values, key=mae_values.get)} (MAE={min(mae_values.values()):.2f})')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Step 6: Analysis and Insights

```python
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

best_method = results_table.index[0]
best_mae = results_table.iloc[0]['MAE']

print(f"\n1. BEST METHOD: {best_method}")
print(f"   - MAE: {best_mae:.2f} passengers")
print(f"   - RMSE: {results_table.iloc[0]['RMSE']:.2f}")
print(f"   - MAPE: {results_table.iloc[0]['MAPE']:.4f} ({results_table.iloc[0]['MAPE']*100:.2f}%)")

print(f"\n2. DATA CHARACTERISTICS:")
# Check if data has strong trend
trend_slope = (test_series[-1] - test_series[0]) / len(test_series)
print(f"   - Trend (slope): {trend_slope:.2f} passengers/month")

# Check if data has strong seasonality
seasonal_naive_mae = evaluation_results.get('Seasonal Naive', {}).get('MAE', np.inf)
naive_mae = evaluation_results.get('Naive', {}).get('MAE', np.inf)
has_seasonality = seasonal_naive_mae < naive_mae
print(f"   - Strong seasonality: {'Yes' if has_seasonality else 'No'}")

# Check noise level
residuals = test_series - all_forecasts[best_method]
noise_std = np.std(residuals)
signal_mean = np.mean(test_series)
snr = signal_mean / noise_std
print(f"   - Signal-to-Noise Ratio: {snr:.2f}")

print(f"\n3. RECOMMENDATIONS FOR NEXT MODULES:")
if has_seasonality:
    print("   - Module 3 (ARIMA): Try SARIMA to capture seasonality")
else:
    print("   - Module 3 (ARIMA): Try ARIMA for trend handling")

if trend_slope > 1:
    print("   - Module 4 (ML): Consider models that capture non-linear trends")
else:
    print("   - Module 4 (ML): Linear models might be sufficient")

print("\n" + "="*60)
```

#### Step 7: Document Your Findings

```python
# Create a summary report
summary_report = f"""
MODULE 2 MINI PROJECT REPORT
============================

Dataset: Airline Passengers (1949-1960)
Train Period: {train_df['Month'].min().date()} to {train_df['Month'].max().date()}
Test Period: {test_df['Month'].min().date()} to {test_df['Month'].max().date()}

BEST PERFORMING METHOD: {best_method}
- MAE: {best_mae:.2f} passengers
- RMSE: {results_table.iloc[0]['RMSE']:.2f} passengers
- MAPE: {results_table.iloc[0]['MAPE']*100:.2f}%

KEY FINDINGS:
1. The data shows strong {('seasonal patterns' if has_seasonality else 'trend')}
2. {best_method} outperformed other baselines by capturing this pattern
3. The average forecast error is about {best_mae:.0f} passengers

NEXT STEPS:
- Module 3 will explore statistical models that can handle both trend and seasonality
- Module 4 will introduce machine learning methods for non-linear patterns
- Baseline methods provide essential comparison points for these advanced approaches

Confidence Level: The test set results show consistent patterns, so we can expect
similar performance on new data with the same characteristics.
"""

print(summary_report)

# Save report
with open('module_2_summary.txt', 'w') as f:
    f.write(summary_report)
```

---

## 2.7 Key Takeaways

✅ **Do This:**
- Start with simple baseline methods every time
- Use proper time-series train-test splits (no shuffling)
- Evaluate using multiple metrics (MAE, RMSE, MAPE)
- Compare methods side-by-side with visualizations
- Document what worked and why

❌ **Don't Do This:**
- Skip baseline methods to jump to complex models
- Randomly shuffle time series data
- Use only one metric to evaluate
- Assume more complex = better
- Ignore the patterns in the data

---

## Progress Checkpoint

**Completion: ~20%** ✓

You've now learned:
- Why baseline methods matter
- How to implement 6 different baseline approaches
- How to properly evaluate forecasts
- How to compare methods systematically

**Next Module (Module 3):** Statistical Time Series Methods
- Learn ARIMA, SARIMA, and Prophet
- Understand stationarity and differencing
- Capture trend and seasonality automatically

**Time Estimate for Module 2:** 6-8 hours
- Reading & understanding: 2-3 hours
- Implementing notebooks: 2-3 hours
- Mini project: 2-3 hours

**Estimated Combined Progress:** 20% → 35%

---

*End of Module 2: Basic Mathematical Methods for Forecasting*
