# Module 6: Advanced Topics in Forecasting

**Duration:** 5-7 hours  
**Prerequisites:** Modules 0-5  
**Learning Level:** Advanced

---

## Table of Contents

1. [Introduction](#introduction)
2. [Multivariate Forecasting](#multivariate)
3. [Anomaly Detection in Time Series](#anomaly)
4. [Change Point Detection](#change-point)
5. [Demand Classification & Category Prediction](#classification)
6. [Prescriptive Forecasting & Optimization](#prescriptive)
7. [Reinforcement Learning for Forecasting](#reinforcement)
8. [Mini-Project 6: Integrated Advanced Analytics](#mini-project)
9. [Production Deployment Strategies](#deployment)
10. [Key Takeaways](#key-takeaways)

---

## 1. Introduction {#introduction}

Advanced forecasting extends beyond simple point predictions. This module covers:

- **Multivariate Models:** Incorporating external variables
- **Anomaly Detection:** Identifying unusual patterns
- **Event Detection:** Finding structural breaks and regime changes
- **Classification:** Predicting categories (high/medium/low demand)
- **Prescriptive Analytics:** Optimizing decisions based on forecasts
- **RL Approaches:** Learning optimal policies from data

---

## 2. Multivariate Forecasting {#multivariate}

### 2.1 Why Multivariate?

Many real-world problems have multiple relevant variables:

```
Sales Forecast depends on:
  - Historical sales
  - Price
  - Marketing spend
  - Seasonality
  - Weather
  - Competitor activity
```

### 2.2 Vector AutoRegression (VAR)

```python
from statsmodels.tsa.api import VAR

def fit_var_model(data, maxlags=2):
    """
    VAR: Each variable predicted from lags of ALL variables
    
    Parameters:
    - data: DataFrame with multiple columns (variables)
    - maxlags: number of lags to include
    """
    model = VAR(data)
    results = model.fit(maxlags, ic='aic')
    
    # Forecast
    forecast = results.get_forecast(steps=12)
    forecast_df = forecast.summary_frame()
    
    return results, forecast_df

# Example:
# data = pd.DataFrame({
#     'sales': [...],
#     'price': [...],
#     'marketing': [...]
# })
# results, forecast = fit_var_model(data)
```

### 2.3 Dynamic Linear Models with Exogenous Variables

```python
from statsmodels.tsa.arima.model import ARIMA

def fit_arimax(endog, exog, order=(1, 1, 1)):
    """
    ARIMA with external regressors (ARIMAX)
    
    endog: endogenous variable (target)
    exog: exogenous variables (external features)
    """
    model = ARIMA(endog, exog=exog, order=order)
    results = model.fit()
    
    return results
```

### 2.4 Neural Networks with Multiple Inputs

```python
import tensorflow as tf
from tensorflow import keras

def build_multivariate_lstm(input_shape, n_features):
    """
    LSTM for multivariate time series
    
    input_shape: (sequence_length,)
    n_features: number of variables (columns)
    """
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True,
                   input_shape=(input_shape, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Single output (target variable)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## 3. Anomaly Detection in Time Series {#anomaly}

### 3.1 Statistical Methods

```python
def detect_anomalies_zscore(data, threshold=3):
    """
    Z-score based anomaly detection
    Points with |z| > threshold are anomalous
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    
    anomalies = z_scores > threshold
    return anomalies, z_scores

def detect_anomalies_iqr(data):
    """
    Interquartile Range (IQR) method
    Robust to outliers
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = (data < lower_bound) | (data > upper_bound)
    return anomalies, lower_bound, upper_bound
```

### 3.2 Isolation Forest

```python
from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(data, contamination=0.1):
    """
    Isolation Forest for anomaly detection
    Uses random forests to isolate anomalies
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    
    # Reshape for sklearn
    X = data.reshape(-1, 1)
    predictions = model.fit_predict(X)
    
    # predictions: -1 for anomalies, 1 for normal
    anomalies = predictions == -1
    
    return anomalies
```

### 3.3 Autoencoder-Based Detection

```python
def build_autoencoder_anomaly_detector(input_shape):
    """
    Autoencoders learn normal patterns
    High reconstruction error = anomaly
    """
    model = keras.Sequential([
        # Encoder
        layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        
        # Decoder
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_shape)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    return model

def detect_anomalies_autoencoder(model, data, threshold=None):
    """
    Detect anomalies using reconstruction error
    """
    # Reconstruct data
    reconstructed = model.predict(data, verbose=0)
    
    # Reconstruction error
    mse = np.mean(np.square(data - reconstructed), axis=1)
    
    # Set threshold if not provided (e.g., 95th percentile)
    if threshold is None:
        threshold = np.percentile(mse, 95)
    
    anomalies = mse > threshold
    
    return anomalies, mse
```

---

## 4. Change Point Detection {#change-point}

### 4.1 CUSUM (Cumulative Sum Control Chart)

```python
def detect_change_points_cusum(data, threshold=5, drift=0):
    """
    CUSUM detects changes in mean
    
    Parameters:
    - threshold: sensitivity (lower = more sensitive)
    - drift: parameter to avoid false positives
    """
    cumsum_pos = np.zeros(len(data))
    cumsum_neg = np.zeros(len(data))
    
    for t in range(1, len(data)):
        mean_val = np.mean(data[:max(10, t)])
        
        cumsum_pos[t] = max(0, cumsum_pos[t-1] + (data[t] - mean_val) - drift)
        cumsum_neg[t] = max(0, cumsum_neg[t-1] - (data[t] - mean_val) - drift)
    
    # Change points where CUSUM exceeds threshold
    change_points = (cumsum_pos > threshold) | (cumsum_neg > threshold)
    
    return change_points, cumsum_pos, cumsum_neg
```

### 4.2 Bayesian Change Point Detection

```python
def detect_change_points_bayesian(data, hazard_rate=1/365):
    """
    Bayesian approach using online learning
    Estimates probability of change at each time point
    """
    n = len(data)
    P = np.zeros(n)  # Probability of change
    
    for t in range(1, n):
        # Prior probability of change
        P_change = hazard_rate
        
        # Posterior probability
        # (Simplified version - full Bayesian model is more complex)
        if t > 1:
            current_val = data[t]
            prev_mean = np.mean(data[max(0, t-30):t])
            prev_std = np.std(data[max(0, t-30):t])
            
            z_score = np.abs((current_val - prev_mean) / (prev_std + 1e-6))
            P[t] = 1 - np.exp(-z_score**2 / 2)
    
    return P
```

---

## 5. Demand Classification & Category Prediction {#classification}

### 5.1 Classification Framework

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def create_demand_categories(data, bins=3):
    """
    Convert continuous forecasts to categories
    
    bins=3: Low (0-33%), Medium (33-66%), High (66-100%)
    """
    categories = pd.qcut(data, q=bins, labels=False)
    return categories

def build_demand_classifier(X, y):
    """
    Classify demand into categories
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X, y)
    
    return model
```

### 5.2 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def build_calibrated_classifier(X, y):
    """
    Output well-calibrated probabilities
    """
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=5
    )
    
    calibrated_model.fit(X, y)
    
    return calibrated_model

def predict_demand_probabilities(model, X_new):
    """
    Get probability for each demand category
    """
    probabilities = model.predict_proba(X_new)
    return probabilities
```

---

## 6. Prescriptive Forecasting & Optimization {#prescriptive}

### 6.1 Optimization with Forecasts

```python
from scipy.optimize import minimize

def optimize_inventory(forecast, holding_cost=1, shortage_cost=10):
    """
    Determine optimal inventory level given demand forecast
    
    Minimize: holding_cost * excess + shortage_cost * deficit
    """
    def objective(inventory_level):
        excess = max(0, inventory_level - forecast)
        deficit = max(0, forecast - inventory_level)
        cost = holding_cost * excess + shortage_cost * deficit
        return cost
    
    # Find optimal level
    result = minimize(
        objective,
        x0=np.mean(forecast),
        bounds=[(0, np.max(forecast) * 2)]
    )
    
    return result.x[0], result.fun

# Example:
# optimal_qty, cost = optimize_inventory(forecast_values)
```

### 6.2 Price Optimization with Demand Forecast

```python
def optimize_price(demand_forecast, cost, elasticity=-1.5):
    """
    Optimize price based on demand forecast
    
    Assume: demand = a * price^elasticity
    """
    def profit_function(price):
        if price <= 0:
            return 0
        
        a = demand_forecast / (cost ** elasticity)
        demand = a * (price ** elasticity)
        
        if demand < 0:
            demand = 0
        
        profit = (price - cost) * demand
        return -profit  # Minimize negative profit
    
    result = minimize(
        profit_function,
        x0=cost * 1.5,
        bounds=[(cost, cost * 5)]
    )
    
    return result.x[0]
```

---

## 7. Reinforcement Learning for Forecasting {#reinforcement}

### 7.1 Basic RL Framework

```python
import gym
import numpy as np
from collections import defaultdict

class ForecastingEnvironment(gym.Env):
    """
    RL environment for learning to forecast
    State: recent data values
    Action: prediction
    Reward: negative error (loss)
    """
    
    def __init__(self, data, lookback=10):
        self.data = data
        self.lookback = lookback
        self.current_idx = lookback
        
    def reset(self):
        self.current_idx = self.lookback
        state = self.data[self.current_idx-self.lookback:self.current_idx]
        return state
    
    def step(self, action):
        # action is the predicted value
        actual = self.data[self.current_idx]
        
        # Reward is negative error
        reward = -np.abs(action - actual)
        
        self.current_idx += 1
        
        done = self.current_idx >= len(self.data)
        
        next_state = self.data[self.current_idx-self.lookback:self.current_idx]
        
        return next_state, reward, done, {}
```

### 7.2 Q-Learning for Discrete Actions

```python
class QLearningForecast:
    """
    Q-learning for discrete demand forecast prediction
    """
    
    def __init__(self, n_bins=10, learning_rate=0.1, discount=0.9):
        self.n_bins = n_bins
        self.lr = learning_rate
        self.discount = discount
        self.Q = defaultdict(lambda: np.zeros(n_bins))
    
    def get_state_key(self, recent_values):
        """Convert continuous state to discrete"""
        return tuple(np.digitize(recent_values, bins=10))
    
    def choose_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)
        
        if np.random.random() < epsilon:
            return np.random.randint(self.n_bins)
        else:
            return np.argmax(self.Q[state_key])
    
    def update_q_value(self, state, action, reward, next_state):
        """Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        old_q = self.Q[state_key][action]
        max_next_q = np.max(self.Q[next_state_key])
        
        new_q = old_q + self.lr * (reward + self.discount * max_next_q - old_q)
        
        self.Q[state_key][action] = new_q
```

---

## 8. Mini-Project 6: Integrated Advanced Analytics {#mini-project}

### Objectives
- Implement anomaly detection on time series
- Detect change points
- Build demand classification model
- Create optimization recommendations
- Synthesize all advanced concepts

### Project Structure

**Part 1: Anomaly Detection**
- Z-score method
- IQR method
- Isolation Forest
- Visualize detected anomalies

**Part 2: Change Point Detection**
- CUSUM algorithm
- Bayesian approach
- Identify structural breaks
- Analyze before/after statistics

**Part 3: Demand Classification**
- Define demand categories (Low/Medium/High)
- Train classifier
- Evaluate classification metrics
- Create probability predictions

**Part 4: Prescriptive Analytics**
- Calculate optimal inventory levels
- Determine optimal pricing
- Cost-benefit analysis
- Recommendations

**Part 5: Synthesis & Reporting**
- Integrated visualizations
- Executive summary
- Actionable recommendations

---

## 9. Production Deployment Strategies {#deployment}

### 9.1 Model Serialization & Versioning

```python
import joblib
import json
from datetime import datetime

def save_model_with_metadata(model, model_name, version, metrics):
    """
    Save model with metadata for tracking
    """
    metadata = {
        'name': model_name,
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model_type': type(model).__name__
    }
    
    # Save model
    joblib.dump(model, f'{model_name}_v{version}.pkl')
    
    # Save metadata
    with open(f'{model_name}_v{version}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def load_model_with_metadata(model_name, version):
    """Load model and verify metadata"""
    model = joblib.load(f'{model_name}_v{version}.pkl')
    
    with open(f'{model_name}_v{version}_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, metadata
```

### 9.2 API Deployment (Flask)

```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load model at startup
model, metadata = load_model_with_metadata('forecast_model', version=1)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predictions
    POST data: {"data": [1, 2, 3, ...]}
    """
    try:
        data = request.json.get('data')
        
        if not data or len(data) != 12:
            return jsonify({'error': 'Invalid input'}), 400
        
        X = np.array(data).reshape(1, -1)
        prediction = model.predict(X)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'model_version': metadata['version'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### 9.3 Monitoring & Retraining

```python
def monitor_model_performance(y_true, y_pred, threshold=0.15):
    """
    Monitor if model performance degrades
    Trigger retraining if MAE % error > threshold
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    needs_retraining = mape > threshold
    
    return {
        'mae': mae,
        'mape': mape,
        'needs_retraining': needs_retraining
    }

def automated_retraining_pipeline(new_data, threshold=0.15):
    """
    Automatically retrain model if needed
    """
    # Current model performance
    y_pred = model.predict(new_data)
    
    monitoring_results = monitor_model_performance(
        y_true=new_data,
        y_pred=y_pred,
        threshold=threshold
    )
    
    if monitoring_results['needs_retraining']:
        print("Retraining triggered due to performance degradation")
        # Retrain logic
        ...
```

---

## 10. Key Takeaways {#key-takeaways}

1. **Multivariate models capture complex relationships** between multiple variables
2. **Anomaly detection prevents models from making extreme predictions**
3. **Change point detection enables model recalibration** after regime shifts
4. **Classification provides actionable categories** for decision-making
5. **Optimization translates forecasts into business decisions**
6. **RL can adapt** to changing patterns through learning
7. **Production systems need monitoring** and automated retraining
8. **Ensemble approach combining multiple advanced techniques** is optimal

---

## Recommended Reading

- Time Series Analysis (Hamilton)
- Demand Forecasting: Evidence-Based Methods and Practical Implementation (Kourentzes)
- Reinforcement Learning: An Introduction (Sutton & Barto)

---

## Next Steps

- Complete [Mini-Project 6](code/module-6-advanced-topics.ipynb)
- Proceed to [Capstone Project](capstone-project.md) for comprehensive integration
- Explore additional resources in course materials

---

**End of Module 6**
