# Module 1: Fundamentals of Forecasting

**Estimated Time:** 4-5 hours  
**Difficulty:** Beginner  
**Prerequisites:** Module 0 (Setup)

---

## Learning Objectives

By the end of this module, you will be able to:
- ✅ Define forecasting and understand its real-world applications
- ✅ Distinguish between time series and non-time series problems
- ✅ Perform exploratory data analysis (EDA) on forecasting data
- ✅ Identify trends, seasonality, and anomalies
- ✅ Decompose time series into components
- ✅ Prepare data for forecasting models
- ✅ Understand train/validation/test splits for time series

---

## 1.1 Why Forecast? Real-World Applications

### 1.1.1 Business Applications

**Retail & E-Commerce**
- Inventory planning: How many units to stock?
- Sales forecasting: Expected revenue next quarter?
- Demand prediction: Which products will be popular?

**Supply Chain**
- Procurement planning: What materials to order?
- Warehouse management: Storage space needed?
- Logistics: Vehicle routing and delivery optimization?

**Finance**
- Stock price prediction: Buy/sell decisions
- Credit risk: Will borrower default?
- Market trends: Economic indicators

**Healthcare**
- Patient volume: Staffing and bed allocation
- Disease spread: Epidemic forecasting
- Supply chain: Medicine and equipment ordering

### 1.1.2 Why Accurate Forecasting Matters
- **Cost Reduction:** Avoid excess inventory (holding costs)
- **Revenue Growth:** Meet demand, increase customer satisfaction
- **Risk Management:** Anticipate problems before they occur
- **Competitive Advantage:** Make faster, data-driven decisions

### 1.1.3 Forecasting Failures (Real Examples)
- **Blockbuster:** Underestimated streaming demand
- **Kodak:** Didn't forecast digital photography adoption
- **Supply Chain:** 2020-2021 bullwhip effect (poor demand forecasting)

---

## 1.2 Types of Forecasting Problems

### 1.2.1 Time Series Forecasting

**Definition:** Predicting a variable that changes over time at regular intervals.

**Key Characteristics:**
- Data has a time index (dates, hours, months)
- Historical values influence future values
- Order matters - cannot shuffle data
- Exhibits trends, seasonality, and cycles

**Examples:**
```
Time Series | Variable | Frequency
-----------|----------|----------
2023-01-01 | Sales    | $10,000
2023-01-02 | Sales    | $12,500
2023-01-03 | Sales    | $11,200
...
2024-01-01 | Sales    | $???? (predict)
```

**Real Examples:**
- Monthly sales (1-12 months ahead)
- Daily stock prices (next week)
- Hourly energy demand (next 24 hours)
- Yearly GDP growth (next fiscal year)

### 1.2.2 Non-Time Series Forecasting (Cross-Sectional)

**Definition:** Predicting outcomes from features without inherent time ordering.

**Key Characteristics:**
- Data points are independent
- Can shuffle data without breaking relationships
- Features determine predictions
- Traditional machine learning approach

**Examples:**
```
Feature 1 | Feature 2 | Feature 3 | Target
----------|-----------|-----------|--------
Age: 25   | Income    | Tenure    | Approved?
Region: NY| Credit    | 5 years   | Yes
...       |           |           |
Age: 35   | Income    | Tenure    | ???? (predict)
```

**Real Examples:**
- Loan approval (customer features → approve/deny)
- Churn prediction (behavior features → will customer leave?)
- Price prediction (house attributes → market value)
- Classification (email → spam or not?)

### 1.2.3 Panel/Mixed Data

**Definition:** Time series data with multiple entities.

**Examples:**
- Sales by store over time (multiple stores, multiple months)
- Temperature by city over time (multiple cities, multiple days)
- Stock prices for multiple companies over time

```
Date       | Store | Sales
-----------|-------|-------
2023-01-01 | NYC   | $5000
2023-01-01 | LA    | $3000
2023-01-02 | NYC   | $5500
2023-01-02 | LA    | $3200
...        | ...   | ...
```

---

## 1.3 Key Forecasting Tasks

### 1.3.1 Regression (Continuous Forecasting)
**Goal:** Predict a continuous numerical value

**Output:** A number (usually with confidence interval)

**Examples:**
- Next month's sales: **$125,000 ± $10,000**
- Tomorrow's temperature: **72°F ± 3°F**
- Stock price in 1 week: **$150.50 ± $5.00**

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### 1.3.2 Classification (Categorical Forecasting)
**Goal:** Predict which category something belongs to

**Output:** A class label with probability

**Examples:**
- Will it rain tomorrow? **Yes (80% confidence)**
- Will customer churn? **No (25% confidence)**
- Demand level next week? **High (60%), Medium (30%), Low (10%)**

**Evaluation Metrics:**
- Accuracy, Precision, Recall
- F1-Score, ROC-AUC

### 1.3.3 Prescriptive Analytics
**Goal:** Recommend optimal actions

**Output:** Action recommendations with expected impact

**Examples:**
- Recommended inventory level: **500 units** (minimizes stockout & holding costs)
- Optimal pricing: **$49.99** (maximizes profit)
- Best staffing level: **15 employees** (meets demand, minimizes labor cost)

**Evaluation:**
- Business KPIs (profit, cost savings, customer satisfaction)

---

## 1.4 Exploratory Data Analysis (EDA) for Forecasting

### 1.4.1 Loading and Inspecting Data

```python
import pandas as pd
import numpy as np

# Load airline passenger data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

# Display first few rows
print(df.head(10))

# Data info
print(df.info())

# Statistical summary
print(df.describe())

# Check for missing values
print(df.isnull().sum())
```

**What to Look For:**
- Data shape (rows, columns)
- Data types (int, float, datetime?)
- Missing values (NaN counts)
- Value ranges (min, max, mean)

### 1.4.2 Time Index Conversion

```python
# Convert 'Month' to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set as index (easier for time series operations)
df = df.set_index('Month')

# Check frequency
df.index.freq
```

### 1.4.3 Basic Visualization

```python
import matplotlib.pyplot as plt

# Simple line plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Passengers'], linewidth=2, color='navy')
plt.title('Airline Passengers (1949-1960)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 1.4.4 Statistical Summary

```python
# Basic statistics
print(f"Mean: {df['Passengers'].mean():.2f}")
print(f"Std Dev: {df['Passengers'].std():.2f}")
print(f"Min: {df['Passengers'].min()}")
print(f"Max: {df['Passengers'].max()}")
print(f"Coefficient of Variation: {df['Passengers'].std() / df['Passengers'].mean():.3f}")
```

---

## 1.5 Key Time Series Characteristics

### 1.5.1 Univariate vs. Multivariate

**Univariate:** One variable to forecast
```
Date       | Sales
-----------|-------
2023-01-01 | 1000
2023-01-02 | 1050
2023-01-03 | 1100
```

**Multivariate:** Multiple predictors or variables
```
Date       | Sales | Advertising | Temperature
-----------|-------|-------------|-------------
2023-01-01 | 1000  | 500         | 65°F
2023-01-02 | 1050  | 600         | 67°F
2023-01-03 | 1100  | 700         | 70°F
```

**When to Use:**
- **Univariate:** Only historical values available or matter
- **Multivariate:** External factors significantly influence target

### 1.5.2 Granularity and Frequency

**Granularity:** Time interval between observations

```
Frequency | Example | Typical Uses
----------|---------|------------------
Second    | Stock trades | High-frequency trading
Minute    | Server metrics | System monitoring
Hour      | Website traffic | Hourly reporting
Day       | Stock prices | Investment decisions
Week      | Sales | Weekly planning
Month     | Revenue | Quarterly reports
Quarter   | GDP | Economic analysis
Year      | Population | Long-term planning
```

**Why It Matters:**
- Finer granularity = more data = more complex patterns
- Coarser granularity = simpler patterns but less prediction power
- Choose based on business decision frequency

### 1.5.3 Trend Analysis

**Definition:** Overall increase or decrease over time

```python
# Visualize trend
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Linear trend
axes[0, 0].plot([1, 2, 3, 4, 5], [10, 12, 14, 16, 18])
axes[0, 0].set_title('Upward Trend')

# Downward trend
axes[0, 1].plot([1, 2, 3, 4, 5], [100, 90, 80, 70, 60])
axes[0, 1].set_title('Downward Trend')

# Non-linear trend
x = [1, 2, 3, 4, 5]
y = [10, 15, 25, 40, 60]
axes[1, 0].plot(x, y)
axes[1, 0].set_title('Non-linear (Exponential) Trend')

# No trend (stationary)
axes[1, 1].plot([1, 2, 3, 4, 5], [50, 51, 50, 49, 50])
axes[1, 1].set_title('Stationary (No Trend)')

plt.tight_layout()
plt.show()
```

### 1.5.4 Seasonality Detection

**Definition:** Regular, repeating patterns at fixed intervals

```python
# Monthly seasonality example
months_data = [100, 120, 110, 130, 125, 140,  # Jan-Jun (Spring/Summer)
               135, 150, 145, 160, 155, 170]  # Jul-Dec (Fall/Winter)

plt.figure(figsize=(10, 5))
plt.plot(months_data, marker='o', linewidth=2)
plt.title('Seasonal Pattern (Higher in Winter)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.grid(True, alpha=0.3)
plt.show()

# Check for seasonality
print(f"Pattern: {months_data[0]} → {months_data[1]} → {months_data[2]}")
print("Notice: Every 12 months repeats similar values")
```

**Common Seasonal Patterns:**
- Daily: Rush hour traffic, hourly temperature patterns
- Weekly: Weekend vs. weekday differences
- Monthly: Beginning/end of month spending patterns
- Yearly: Holiday shopping, summer vacations, tax season
- Quarterly: Fiscal quarter effects

### 1.5.5 Autocorrelation

**Definition:** Correlation of a series with its past values

```python
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
autocorrelation_plot(df['Passengers'])
plt.title('Autocorrelation: How Much Past Values Influence Future')
plt.tight_layout()
plt.show()
```

**Interpretation:**
- High autocorrelation = past values strongly predict future (good!)
- Low autocorrelation = harder to forecast (more randomness)
- Regular spikes = seasonality at that lag

---

## 1.6 Time Series Decomposition

### 1.6.1 Components of a Time Series

Every time series can be decomposed into:

```
Y(t) = Trend + Seasonality + Residual

Example: Sales = Long-term growth + Holiday spikes + Random noise
```

### 1.6.2 Decomposition Example

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
decomposition = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Original data
axes[0].plot(decomposition.observed, color='navy')
axes[0].set_ylabel('Observed')
axes[0].set_title('Time Series Decomposition')

# Trend
axes[1].plot(decomposition.trend, color='green')
axes[1].set_ylabel('Trend')

# Seasonality
axes[2].plot(decomposition.seasonal, color='orange')
axes[2].set_ylabel('Seasonal')

# Residual
axes[3].plot(decomposition.resid, color='red')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.show()
```

**What Each Component Means:**
- **Trend:** Long-term direction (general growth/decline)
- **Seasonality:** Regular, repeating patterns
- **Residual:** Random fluctuations unexplained by trend/seasonality

---

## 1.7 Train/Validation/Test Splits for Time Series

### 1.7.1 Why NOT Shuffle?

**Cross-sectional data (OK to shuffle):**
```
Shuffle ✓: [Sample 1] [Sample 3] [Sample 2] [Sample 5] [Sample 4]
```

**Time Series (DO NOT shuffle):**
```
Future →
Time --------→
|-------- Train --------|-- Valid --|-- Test --|

Wrong ✗: [Week 3] [Week 1] [Week 4] [Week 2] [Week 5]
         (Information leakage!)

Right ✓: [Week 1] [Week 2] [Week 3] | [Week 4] | [Week 5]
```

### 1.7.2 Time Series Split Strategy

```python
# 70% Train, 15% Validation, 15% Test
n = len(df)
train_size = int(0.70 * n)
val_size = int(0.15 * n)

df_train = df[:train_size]
df_val = df[train_size:train_size + val_size]
df_test = df[train_size + val_size:]

print(f"Train: {len(df_train)} samples")
print(f"Validation: {len(df_val)} samples")
print(f"Test: {len(df_test)} samples")

# Visualize the split
plt.figure(figsize=(14, 5))
plt.plot(df_train.index, df_train['Passengers'], label='Train', color='blue')
plt.plot(df_val.index, df_val['Passengers'], label='Validation', color='orange')
plt.plot(df_test.index, df_test['Passengers'], label='Test', color='red')
plt.legend()
plt.title('Time Series Train/Val/Test Split')
plt.show()
```

---

## 1.8 Mini Project: Comprehensive Dataset Exploration

### 1.8.1 Project Overview
- **Goal:** Develop deep intuition about time series data through hands-on exploration
- **Duration:** 2-3 hours
- **Deliverable:** Jupyter notebook with EDA, visualizations, and written insights
- **Grading:** Completeness and depth of analysis

### 1.8.2 Project Steps (Detailed)

**Step 1: Choose Your Dataset**
- Use provided airline passenger data, OR
- Pick from: [Kaggle](https://www.kaggle.com/datasets), [UCI](https://archive.ics.uci.edu/), [FRED](https://fred.stlouisfed.org/)
- Recommended datasets:
  - Energy consumption (hourly/daily)
  - Stock prices (daily)
  - Web traffic (daily)
  - Weather data (hourly)
  - Sales/retail data (daily/weekly)

**Step 2: Load and Inspect**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('your_data.csv')

# Answer these questions:
print("1. Data Shape:", df.shape)
print("2. Columns:", df.columns.tolist())
print("3. Data Types:")
print(df.dtypes)
print("4. Missing Values:")
print(df.isnull().sum())
print("5. First 10 rows:")
print(df.head(10))
print("6. Last 10 rows:")
print(df.tail(10))
```

**Step 3: Time Series Setup**
```python
# Convert date column if needed
df['Date'] = pd.to_datetime(df['Date'])

# Set as index
df = df.set_index('Date')

# Check frequency
print(f"Frequency: {pd.infer_freq(df.index)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Total days: {(df.index.max() - df.index.min()).days}")
```

**Step 4: Exploratory Data Analysis (5+ visualizations)**

Visualization 1: Raw Time Series
```python
plt.figure(figsize=(14, 5))
plt.plot(df.index, df['Target'], linewidth=1.5)
plt.title('Raw Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

Visualization 2: Statistical Summary
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histogram
axes[0, 0].hist(df['Target'], bins=50, edgecolor='black')
axes[0, 0].set_title('Distribution of Values')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Box plot
axes[0, 1].boxplot(df['Target'])
axes[0, 1].set_title('Box Plot (Outliers & Quartiles)')

# Rolling mean
rolling_mean = df['Target'].rolling(window=30).mean()
axes[1, 0].plot(df.index, df['Target'], alpha=0.5, label='Original')
axes[1, 0].plot(df.index, rolling_mean, color='red', label='30-period Rolling Mean')
axes[1, 0].set_title('Trend (30-day Moving Average)')
axes[1, 0].legend()

# Seasonal subseries
if len(df) >= 365:
    df['DayOfYear'] = df.index.dayofyear
    axes[1, 1].scatter(df['DayOfYear'], df['Target'], alpha=0.3)
    axes[1, 1].set_title('Seasonality Check (by day of year)')
    axes[1, 1].set_xlabel('Day of Year')
    axes[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.show()
```

Visualization 3: Autocorrelation
```python
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(12, 5))
autocorrelation_plot(df['Target'], lags=50)
plt.title('Autocorrelation (How Past Values Predict Future)')
plt.tight_layout()
plt.show()
```

Visualization 4: Decomposition (if applicable)
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Only works if sufficient data with clear seasonality
if len(df) >= 104:  # At least 2 years
    decomposition = seasonal_decompose(df['Target'], period=52)  # Weekly seasonality
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

Visualization 5: Monthly/Seasonal Patterns
```python
# If monthly data
df['Month'] = df.index.month
monthly_avg = df.groupby('Month')['Target'].mean()

plt.figure(figsize=(10, 5))
plt.bar(monthly_avg.index, monthly_avg.values, color='steelblue', edgecolor='black')
plt.title('Average Value by Month (Seasonality Check)')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

**Step 5: Answer Critical Questions (Written Analysis)**

In your notebook, answer each with evidence from your data:

1. **What are the characteristics of your dataset?**
   - Size: How many observations?
   - Time span: What period does it cover?
   - Frequency: Daily? Monthly? Other?
   - Type: Is it time series or cross-sectional?

2. **What are you trying to forecast?**
   - Which variable is your target?
   - Is prediction practical/useful?
   - What decisions would be made with accurate forecasts?

3. **Does your data show a trend?**
   - Generally increasing, decreasing, or flat?
   - Linear or non-linear?
   - How strong is the trend?
   - Quote specific values/dates as evidence

4. **Does your data show seasonality?**
   - Do patterns repeat at fixed intervals?
   - If yes, what is the period (daily, weekly, monthly, yearly)?
   - How strong is the seasonal effect?
   - Provide specific examples

5. **Are there anomalies or outliers?**
   - Unexplained spikes or drops?
   - Missing data or gaps?
   - Data quality issues?
   - How will you handle them?

6. **What external factors might influence your data?**
   - Holidays? Marketing campaigns? Product launches?
   - Weather? Economic conditions? Regulatory changes?
   - How might these appear in the data?

7. **What modeling approaches seem appropriate?**
   - Stationary or non-stationary? (needs differencing?)
   - Strong seasonality? (needs seasonal methods?)
   - External variables available? (needs exogenous features?)
   - Hypothesis: What model type might work best?

**Step 6: Document Findings**

Create a summary section in your notebook:

```python
# ============================================
# SUMMARY OF FINDINGS
# ============================================

findings = """
DATASET CHARACTERISTICS:
- Size: X observations over Y period
- Frequency: Daily/Weekly/Monthly
- Target variable: [Variable name]

PATTERNS OBSERVED:
- Trend: [Describe, with evidence]
- Seasonality: [Describe, with evidence]
- Anomalies: [List any notable events]

KEY INSIGHTS:
1. [Insight about the data]
2. [Insight about forecasting challenges]
3. [Insight about data quality]

NEXT STEPS FOR MODELING:
- Consider [Method 1] because...
- May need [Preprocessing] for...
- Should validate with [Validation strategy] to...
"""

print(findings)
```

---

## 1.9 Data Preparation Checklist

Before moving to Module 2, ensure:

- [ ] Data is loaded and indexed by time
- [ ] Data types are correct (datetime, numeric)
- [ ] Missing values handled (documented)
- [ ] Outliers identified and understood
- [ ] Trends identified and described
- [ ] Seasonality identified (if exists)
- [ ] Train/Val/Test split created
- [ ] No future information leakage

---

## 1.10 Common Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Missing values | Interpolate (linear, forward fill) or document |
| Gaps in time series | Resample to regular frequency |
| Outliers | Document cause; keep real events |
| Non-stationary | Note for Module 3 (will address) |
| Seasonal periods unclear | Check ACF plot, domain knowledge |
| Insufficient data | Consider simpler models, external data |

---

## 1.11 Key Takeaways

- ✅ Time series has **temporal order** - don't shuffle!
- ✅ **EDA is crucial** - understand your data before modeling
- ✅ Look for **trends, seasonality, and anomalies**
- ✅ **Decomposition helps** visualize time series components
- ✅ **Train/Val/Test must respect time order**
- ✅ External factors matter - consider them in your analysis

---

## 1.12 Knowledge Check

Before moving to Module 2, verify you can answer:

1. What's the difference between trend and seasonality?
2. Why can't you shuffle time series data?
3. How do you check for trends in your data?
4. What does a high autocorrelation value mean?
5. Why is train/val/test split important for forecasting?

---

## 1.13 Next Steps

✅ **Module 1 Complete!**

**You're ready for Module 2:** [Basic Mathematical Methods](module-2-basic-mathematical-methods.md)

**Module 2 Preview:**
- Naive forecasting methods (baselines)
- Moving averages and exponential smoothing
- Simple regression approaches
- Performance evaluation metrics

---

*Module 1 Complete*  
**Total Course Progress:** 13% (1/8 modules)  
**Time Invested:** ~4-5 hours  
**Next Module Time:** 6-8 hours