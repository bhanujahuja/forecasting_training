# Module 1: Fundamentals of Forecasting

## 1.1 Why Forecast? Real-world Examples

- Inventory management, financial markets, energy load, weather, demand, classification (will it rain? will customer churn?)  
- *Project kickoff:* Find your own use case or pick one of the datasets from above examples.

---

## 1.2 Types of Forecasting Problems

- **Time Series**  
  Data ordered in time.  
  *Examples:* Temperature by hour, sales by day.
- **Cross-sectional (non-time-series)**  
  Data not ordered by time.  
  *Examples:* Predicting product demand for next month from features.
- **Panel/Mixed Data**  
  Data with both time & other dimensions (e.g., sales by region over months).

---

## 1.3 Key Forecasting Tasks

- **Prediction/Regression:** Continuous value forecast (next day's sales).
- **Classification:** Categorical outcome forecast (will it rain, yes/no?).
- **Prescriptive:** Making decisions based on forecast (how much to stock?).

---

## 1.4 Exploratory Data Analysis (EDA) for Forecasting

### 1.4.1 Data Loading Example
```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
print(df.head())
```

### 1.4.2 Checking for Time Series Properties

```python
print(df.dtypes)
print(df['Month'].head())
```

### 1.4.3 Visualizing Your Data

```python
import matplotlib.pyplot as plt
plt.plot(df['Month'], df['Passengers'])
plt.title("Airline Passengers Over Time")
plt.xticks(rotation=45)
plt.show()
```

---

## 1.5 Characteristics of Forecasting Data

- **Univariate vs. Multivariate**
    - Univariate: one variable to forecast (sales).
    - Multivariate: multiple predictors (ad spend, season, etc.).
- **Granularity & Periodicity**
    - Hourly, daily, weekly, monthly
- **Trends & Seasonality**
    - Trend = overall increase/decrease
    - Seasonality = cycles (e.g., weekends, holidays)

---

## 1.6 Key Concepts

- **Training/Validation/Test Splits in Forecasting**
    - Why not shuffle? Respect time order!
- **Overfitting & Underfitting**

---

## 1.7 Mini Project: Explore and Visualize a Dataset

### 1.7.1 Project Overview
- **Goal:** Develop intuition by exploring real-world time series data
- **Deliverable:** Jupyter notebook with EDA, plots, and insights

### 1.7.2 Project Steps

1. **Choose your dataset**
   - Use airline passengers (provided), or pick from Kaggle/UCI
   - Options: Energy consumption, stock prices, weather, sales data
   
2. **Load and inspect the data**
   ```python
   import pandas as pd
   df = pd.read_csv("your_data.csv")
   print(df.info())
   print(df.describe())
   ```

3. **Identify key characteristics**
   - Time series or non-time-series?
   - Granularity (hourly, daily, weekly, etc.)
   - Missing values? Duplicates?
   - Data types and ranges

4. **Exploratory Data Analysis (EDA)**
   - Visualize raw data (line plot)
   - Check for trends (increasing/decreasing patterns)
   - Identify seasonality (repeating cycles)
   - Analyze statistical properties (mean, std, skewness)

5. **Answer these critical questions**
   - What are you trying to forecast?
   - Is data time series?
   - What patterns (trend/seasonality) do you observe?
   - Are there obvious outliers or anomalies?
   - What external factors might influence the data?

6. **Document findings**
   - Create summary statistics
   - Include 4-5 visualizations
   - Write conclusions about data characteristics

---

## 1.8 Common Challenges & Tips

- **Missing data:** Use interpolation or forward fill methods
- **Outliers:** Document but don't remove; some are real events
- **Non-stationary data:** Differences are expected; will address in Module 3
- **Data quality:** Check for obvious errors (negative sales, etc.)

---

## 1.9 Next Up

- Mathematical/basic forecasting methods  
  (see Module 2 in outline)

---

*End of Module 1*