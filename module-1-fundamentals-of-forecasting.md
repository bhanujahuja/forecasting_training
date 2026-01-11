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

- Pick a dataset, load it, and create first plots as above.
- Answer these:
    - What are you forecasting?
    - Is data time series?
    - What trends or seasonality do you see?
- Share your notebook in the course repo!

---

## 1.8 Next Up

- Mathematical/basic forecasting methods  
  (see Module 2 in outline)

---

*End of Module 1*