# Module 4: Machine Learning for Forecasting

**Duration:** 10-12 hours (including mini-project)  
**Prerequisites:** Modules 0-3, familiarity with scikit-learn  
**Learning Level:** Intermediate to Advanced

---

## Table of Contents

1. [Introduction](#introduction)
2. [Feature Engineering for Time Series](#feature-engineering)
3. [Supervised Learning Framework](#supervised-framework)
4. [ML Model Implementations](#ml-models)
5. [Time-Aware Cross-Validation](#cross-validation)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Mini-Project 4: ML Forecasting Pipeline](#mini-project)
8. [Common Challenges & Solutions](#challenges)
9. [Key Takeaways](#key-takeaways)

---

## 1. Introduction {#introduction}

While statistical methods (ARIMA, SARIMA, Prophet) excel at capturing temporal patterns, **machine learning approaches** offer distinct advantages:

### Why Machine Learning for Forecasting?

| Aspect | Statistical | Machine Learning |
|--------|-------------|------------------|
| **Multiple Features** | Limited | ✓ Excellent |
| **Non-linear Patterns** | Limited | ✓ Excellent |
| **Large Datasets** | Slow | ✓ Fast |
| **Interpretability** | High | Medium |
| **Requires Stationarity** | Yes | No |
| **Hyperparameter Tuning** | Limited | Extensive |

### Key Difference: Problem Framing

- **Statistical:** Time series → univariate or multivariate regression
- **ML:** Time series → **Supervised learning problem** (with engineered features)

---

## 2. Feature Engineering for Time Series {#feature-engineering}

Converting time series into supervised learning requires creating **features** and **targets**:

### 2.1 Lag Features

Create historical values as features:

```python
import pandas as pd
import numpy as np

def create_lag_features(data, lags=[1, 3, 6, 12]):
    """
    Create lagged features from univariate time series
    
    Parameters:
    - data: pd.Series or pd.DataFrame with single column
    - lags: list of lag periods
    
    Returns:
    - DataFrame with target and lag features
    """
    df = pd.DataFrame({'y': data})
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['target'] = df['y'].shift(-1)  # Next period forecast
    return df.dropna()

# Example:
# Original series: [10, 15, 12, 18, 20, 22]
# Creates: lag_1, lag_3, lag_6 features + target (next value)
```

### 2.2 Rolling Statistics

Capture short-term trends and volatility:

```python
def create_rolling_features(data, windows=[3, 6, 12]):
    """
    Create rolling statistical features
    """
    df = pd.DataFrame({'y': data})
    
    for w in windows:
        df[f'rolling_mean_{w}'] = df['y'].rolling(w).mean()
        df[f'rolling_std_{w}'] = df['y'].rolling(w).std()
        df[f'rolling_min_{w}'] = df['y'].rolling(w).min()
        df[f'rolling_max_{w}'] = df['y'].rolling(w).max()
    
    df['target'] = df['y'].shift(-1)
    return df.dropna()
```

### 2.3 Seasonality Indicators

Capture repeating patterns:

```python
def create_seasonality_features(dates, period=12):
    """
    Create seasonal indicators (month, quarter, day-of-week, etc)
    """
    df = pd.DataFrame({'date': dates})
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Create sine/cosine encoding (circular nature)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df
```

### 2.4 Trend Features

Capture long-term direction:

```python
def create_trend_features(data):
    """
    Create trend indicators
    """
    df = pd.DataFrame({'y': data})
    df['index'] = range(len(df))
    
    # Linear trend
    from sklearn.linear_model import LinearRegression
    X = df[['index']].values
    y = df['y'].values
    model = LinearRegression()
    model.fit(X, y)
    df['trend'] = model.predict(X)
    
    return df
```

### Complete Feature Engineering Example

```python
def engineer_features(data, date_index=None, lags=[1, 3, 6, 12], 
                      rolling_windows=[3, 6, 12]):
    """
    Comprehensive feature engineering pipeline
    """
    # Start with lags and rolling stats
    df = create_lag_features(data, lags)
    df = df.drop('target', axis=1)
    
    # Add rolling features
    for w in rolling_windows:
        df[f'rolling_mean_{w}'] = data.rolling(w).mean()
        df[f'rolling_std_{w}'] = data.rolling(w).std()
    
    # Add seasonality if dates provided
    if date_index is not None:
        seasonals = create_seasonality_features(date_index)
        df = pd.concat([df, seasonals], axis=1)
    
    # Target: next period value
    df['target'] = data.shift(-1)
    
    return df.dropna()
```

---

## 3. Supervised Learning Framework {#supervised-framework}

### 3.1 Train-Test Split for Time Series

**Critical:** Never shuffle time series data!

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_train_test_split(X, y, test_size=0.2):
    """
    Proper time-series split (no shuffling)
    """
    split_point = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    return X_train, X_test, y_train, y_test

# Alternative: TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate model
```

### 3.2 Scaling for ML Models

Most ML algorithms benefit from feature scaling:

```python
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    """
    Fit scaler on training data, apply to test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
```

---

## 4. ML Model Implementations {#ml-models}

### 4.1 Linear Regression

**Best for:** Linear trends, few features

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def fit_linear_regression(X_train, y_train, X_test, y_test):
    """
    Baseline linear model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, {'MAE': mae, 'RMSE': rmse}, y_pred
```

### 4.2 Random Forest

**Best for:** Non-linear patterns, feature importance

```python
from sklearn.ensemble import RandomForestRegressor

def fit_random_forest(X_train, y_train, X_test, y_test, 
                      n_estimators=100, max_depth=10):
    """
    Random Forest for non-linear forecasting
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, {'MAE': mae, 'RMSE': rmse}, y_pred, feature_importance
```

### 4.3 Gradient Boosting (XGBoost, LightGBM)

**Best for:** Complex patterns, competitions

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error

def fit_xgboost(X_train, y_train, X_test, y_test, 
                learning_rate=0.1, max_depth=5, n_estimators=200):
    """
    XGBoost for advanced forecasting
    """
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42,
        eval_metric='mae'
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred)
    }
    
    return model, metrics, y_pred
```

### 4.4 Support Vector Regression

**Best for:** Small datasets, high-dimensional data

```python
from sklearn.svm import SVR

def fit_svr(X_train, y_train, X_test, y_test, kernel='rbf', C=100):
    """
    Support Vector Regression
    """
    model = SVR(kernel=kernel, C=C, epsilon=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, {'MAE': mae, 'RMSE': rmse}, y_pred
```

---

## 5. Time-Aware Cross-Validation {#cross-validation}

Never use standard K-fold CV for time series!

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(X, y, model, cv_splits=5):
    """
    Time-aware cross-validation
    Only trains on past data, tests on future data
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        cv_scores.append({'MAE': mae, 'RMSE': rmse})
    
    # Average scores
    avg_scores = {
        'MAE': np.mean([s['MAE'] for s in cv_scores]),
        'RMSE': np.mean([s['RMSE'] for s in cv_scores])
    }
    
    return avg_scores, cv_scores
```

---

## 6. Hyperparameter Optimization {#hyperparameter-optimization}

### 6.1 Grid Search with Time Series CV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def grid_search_time_series(X, y, param_grid, model_class,
                            cv_splits=5):
    """
    Grid search with time series cross-validation
    """
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_class())
    ])
    
    # Grid search
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    return grid.best_params_, grid.best_score_, grid
```

### 6.2 Randomized Search (for large parameter spaces)

```python
from sklearn.model_selection import RandomizedSearchCV

def randomized_search_time_series(X, y, param_dist, model_class,
                                  n_iter=20, cv_splits=5):
    """
    Randomized search for large hyperparameter spaces
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_class())
    ])
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    return random_search.best_params_, random_search.best_score_
```

---

## 7. Mini-Project 4: ML Forecasting Pipeline {#mini-project}

### Objectives
- Engineer comprehensive features from time series
- Implement 5+ ML models
- Perform time-aware model selection
- Compare ML with statistical methods from Module 3
- Create visualizations and summary

### Project Structure

**Part 1: Feature Engineering**
- Create lag features (1, 3, 6, 12 periods)
- Add rolling statistics (mean, std, min, max)
- Include seasonal indicators (month, quarter, sin/cos encoding)
- Implement trend features

**Part 2: Model Implementation**
- Linear Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Support Vector Regression (optional)

**Part 3: Model Evaluation**
- Time-series train-test split (80-20)
- Time-aware cross-validation
- Multi-metric comparison (MAE, RMSE, MAPE)
- Feature importance analysis
- Residual diagnostics

**Part 4: Comparison & Visualization**
- ML vs Statistical methods (Module 3)
- Model complexity vs performance tradeoff
- Forecast plots with confidence intervals
- Feature importance barplot
- Residual analysis

**Part 5: Insights & Recommendations**
- Model selection rationale
- Computational efficiency analysis
- Production readiness assessment
- Recommendations for deployment

---

## 8. Common Challenges & Solutions {#challenges}

| Challenge | Solution |
|-----------|----------|
| **Overfitting on small data** | Use regularization, ensemble methods, increase training data |
| **Temporal leakage** | Ensure proper time-series CV, no future info in features |
| **Slow training** | Reduce features, subsample data, use LightGBM instead of XGBoost |
| **Poor generalization** | More features, better CV strategy, ensemble methods |
| **Feature scaling issues** | Always fit scaler on training data only |
| **Imbalanced time periods** | Use stratified sampling or weighted loss |

---

## 9. Key Takeaways {#key-takeaways}

1. **Feature engineering is critical:** ML success depends 80% on feature quality
2. **Time-aware validation is essential:** Never shuffle time series data
3. **Ensemble methods work best:** Combine multiple models for robustness
4. **Hyperparameter tuning matters:** Grid search with proper CV
5. **Compare with statistical methods:** ML is not always better than ARIMA
6. **Interpretability matters:** Understand why your model makes predictions
7. **Scalability:** ML methods scale better with large datasets

---

## Next Steps

- Complete [Mini-Project 4](code/module-4-machine-learning-for-forecasting.ipynb)
- Review [Module 5: Deep Learning](module-5-deep-learning-and-ai-methods.md) for neural network approaches
- Explore hyperparameter tuning strategies in depth

---

**End of Module 4**
