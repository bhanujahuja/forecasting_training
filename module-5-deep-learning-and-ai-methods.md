# Module 5: Deep Learning & AI Methods for Forecasting

**Duration:** 10-12 hours (including mini-project)  
**Prerequisites:** Modules 0-4, basic neural network knowledge  
**Learning Level:** Advanced

---

## Table of Contents

1. [Introduction to Deep Learning](#introduction)
2. [Feedforward Neural Networks](#feedforward)
3. [Recurrent Neural Networks & LSTM](#rnn-lstm)
4. [1D Convolutional Neural Networks](#cnn-1d)
5. [Attention Mechanisms](#attention)
6. [Hybrid & Ensemble Approaches](#hybrid-ensemble)
7. [Mini-Project 5: Deep Learning Forecast](#mini-project)
8. [Practical Considerations & Deployment](#deployment)
9. [Key Takeaways](#key-takeaways)

---

## 1. Introduction to Deep Learning {#introduction}

Deep learning methods excel at capturing **complex temporal dependencies** and **non-linear patterns** in time series data. Unlike traditional ML methods, neural networks can learn hierarchical feature representations.

### Why Deep Learning for Forecasting?

| Advantage | Description |
|-----------|-------------|
| **Automatic Feature Learning** | Learns features without explicit engineering |
| **Captures Long-Range Dependencies** | Especially with LSTM/GRU architectures |
| **Flexible Architecture** | Can model complex interactions |
| **Transfer Learning** | Pre-trained models for similar domains |
| **Ensemble-Friendly** | Easy to combine multiple architectures |

### Disadvantages

| Limitation | Mitigation |
|-----------|-----------|
| **Requires Large Data** | Use transfer learning or regularization |
| **Computational Cost** | Use smaller models or reduce history window |
| **Hyperparameter Tuning** | Use systematic grid/random search |
| **Black Box Nature** | Use SHAP/LIME for interpretability |
| **Overfitting Risk** | Dropout, batch norm, early stopping |

---

## 2. Feedforward Neural Networks {#feedforward}

### 2.1 Architecture

```
Input Layer (features)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit, Linear)
```

### 2.2 Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feedforward_model(input_shape, learning_rate=0.001):
    """
    Build a feedforward neural network for forecasting
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model
```

### 2.3 Training with Callbacks

```python
def train_feedforward_model(model, X_train, y_train, X_val, y_val,
                           epochs=100, batch_size=32):
    """
    Train with early stopping and learning rate reduction
    """
    
    callbacks = [
        # Stop if validation loss doesn't improve
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # Reduce learning rate if stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

---

## 3. Recurrent Neural Networks & LSTM {#rnn-lstm}

### 3.1 LSTM Architecture

**Long Short-Term Memory (LSTM)** networks are designed for sequential data:

```
Input Sequence
    ↓
LSTM Layer (units=64, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (units=32, return_sequences=False)
    ↓
Dropout (0.2)
    ↓
Dense Layer (16 units, ReLU)
    ↓
Output Layer (1 unit)
```

### 3.2 LSTM Implementation

```python
def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Build LSTM model for time series forecasting
    
    input_shape: (sequence_length, n_features)
    """
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(64, activation='relu', return_sequences=True, 
                   input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

### 3.3 Data Preparation for Sequences

```python
def create_sequences(data, lookback=12):
    """
    Create sequences for LSTM input
    
    Parameters:
    - data: 1D array of time series values
    - lookback: number of past timesteps (sequence length)
    
    Returns:
    - X: (n_samples, lookback, 1)
    - y: (n_samples,)
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)

# Example:
# lookback=12 means use 12 past months to predict next month
X, y = create_sequences(data, lookback=12)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)
```

### 3.4 Bidirectional LSTM (Optional)

```python
def build_bidirectional_lstm(input_shape, learning_rate=0.001):
    """
    Bidirectional LSTM: reads sequence forward AND backward
    More powerful for capturing patterns but slower
    """
    model = keras.Sequential([
        layers.Bidirectional(
            layers.LSTM(64, activation='relu', return_sequences=True),
            input_shape=input_shape
        ),
        layers.Dropout(0.2),
        layers.Bidirectional(
            layers.LSTM(32, activation='relu', return_sequences=False)
        ),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## 4. 1D Convolutional Neural Networks {#cnn-1d}

### 4.1 Why Conv1D for Time Series?

- **Captures local temporal patterns** (sliding window)
- **Parameter efficient** (fewer weights than dense layers)
- **Fast inference** (highly optimized)
- **Multi-scale learning** (multiple filter sizes)

### 4.2 Conv1D Architecture

```
Input (sequence)
    ↓
Conv1D (filters=64, kernel_size=3) → ReLU
    ↓
Conv1D (filters=32, kernel_size=3) → ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Flatten
    ↓
Dense (16) → ReLU
    ↓
Output
```

### 4.3 Implementation

```python
def build_cnn_1d_model(input_shape, learning_rate=0.001):
    """
    Build 1D CNN model for time series
    """
    model = keras.Sequential([
        # First conv block
        layers.Conv1D(64, kernel_size=3, activation='relu',
                     input_shape=input_shape, padding='same'),
        layers.Dropout(0.2),
        
        # Second conv block
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Third conv block
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## 5. Attention Mechanisms {#attention}

### 5.1 Self-Attention

Allows model to focus on relevant timesteps:

```python
def build_attention_lstm(input_shape, learning_rate=0.001):
    """
    LSTM with attention mechanism
    """
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
    lstm_out = layers.LSTM(32, return_sequences=True)(lstm_out)
    
    # Attention
    attention = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=16
    )(lstm_out, lstm_out)
    
    # Output
    output = layers.GlobalAveragePooling1D()(attention)
    output = layers.Dense(16, activation='relu')(output)
    output = layers.Dense(1)(output)
    
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

---

## 6. Hybrid & Ensemble Approaches {#hybrid-ensemble}

### 6.1 CNN-LSTM Hybrid

```python
def build_cnn_lstm_model(input_shape, learning_rate=0.001):
    """
    Combine CNN and LSTM benefits:
    - CNN learns local patterns
    - LSTM models long-range dependencies
    """
    model = keras.Sequential([
        # CNN for feature extraction
        layers.Conv1D(64, kernel_size=3, activation='relu',
                     input_shape=input_shape, padding='same'),
        layers.Dropout(0.2),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Dropout(0.2),
        
        # LSTM for temporal dependency
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
```

### 6.2 Ensemble of Models

```python
def build_ensemble_forecast(X_test, models_dict):
    """
    Combine predictions from multiple models
    
    models_dict: {model_name: trained_model}
    """
    predictions = {}
    
    for name, model in models_dict.items():
        pred = model.predict(X_test, verbose=0)
        predictions[name] = pred
    
    # Average ensemble (can also use weighted ensemble)
    ensemble_pred = np.mean(
        list(predictions.values()),
        axis=0
    )
    
    return ensemble_pred, predictions
```

---

## 7. Mini-Project 5: Deep Learning Forecast {#mini-project}

### Objectives
- Build and train multiple DL architectures
- Implement sequence data preparation
- Compare DL vs ML vs Statistical methods
- Create comprehensive visualizations
- Evaluate model uncertainty

### Project Structure

**Part 1: Data Preparation**
- Create sequences with lookback window
- Normalize/scale features (0-1 range)
- Split into train/validation/test sets
- Visualize sequence structure

**Part 2: Model Building & Training**
- Feedforward Neural Network
- LSTM (single and stacked)
- 1D CNN
- Hybrid CNN-LSTM
- Train all models with early stopping

**Part 3: Evaluation**
- Multi-metric comparison (MAE, RMSE, MAPE)
- Training history visualization
- Convergence analysis
- Generalization assessment

**Part 4: Prediction & Visualization**
- Generate forecasts
- Plot actual vs predicted
- Create ensemble predictions
- Confidence intervals (bootstrap)

**Part 5: Comparative Analysis**
- DL vs ML vs Statistical methods
- Computational cost comparison
- Model complexity vs performance
- Final recommendations

---

## 8. Practical Considerations & Deployment {#deployment}

### 8.1 Data Normalization

```python
from sklearn.preprocessing import MinMaxScaler

def scale_time_series(data, scaler_range=(0, 1)):
    """
    Scale time series to [0, 1] for neural networks
    """
    scaler = MinMaxScaler(feature_range=scaler_range)
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Always fit scaler on training data only
X_train_scaled, scaler = scale_time_series(X_train)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
```

### 8.2 Batch Size Selection

```python
# Guidance:
batch_size_options = {
    'small_data (< 1000)': 16,
    'medium_data (1000-10000)': 32,
    'large_data (> 10000)': 64,
    'very_large_data (> 100000)': 128
}

# Smaller batch = more gradient updates (better generalization)
# Larger batch = faster training, but less frequent updates
```

### 8.3 Early Stopping

```python
keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=10,         # Stop if no improvement for 10 epochs
    restore_best_weights=True,  # Use best weights
    min_delta=0.001      # Minimum improvement threshold
)
```

### 8.4 Model Serialization

```python
# Save trained model
model.save('my_forecast_model.h5')

# Load model
loaded_model = keras.models.load_model('my_forecast_model.h5')

# Make predictions with loaded model
predictions = loaded_model.predict(X_test)
```

### 8.5 Uncertainty Quantification

```python
def predict_with_uncertainty(model, X_test, n_iterations=100):
    """
    Estimate prediction uncertainty using Monte Carlo dropout
    """
    predictions = []
    
    for _ in range(n_iterations):
        # Enable dropout during inference
        pred = model(X_test, training=True)
        predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

---

## 9. Key Takeaways {#key-takeaways}

1. **LSTM excels at capturing long-term dependencies** in time series
2. **CNN is parameter-efficient** for local pattern learning
3. **Hybrid architectures** (CNN-LSTM) often perform best
4. **Data normalization is critical** for neural network convergence
5. **Early stopping prevents overfitting** in deep learning
6. **Ensemble methods improve robustness** across different architectures
7. **Deep learning shines with large, complex datasets**
8. **Statistical + ML + DL ensemble** is often optimal strategy

---

## Architectural Recommendations

### For Simple Time Series
- Linear Regression or Statistical Methods (ARIMA)

### For Moderate Complexity
- Random Forest or XGBoost (Module 4)
- LSTM (single layer, short sequence)

### For Complex Patterns
- Hybrid CNN-LSTM
- Stacked LSTM with attention
- Ensemble of DL models

### For Real-Time Systems
- 1D CNN (fast inference)
- Lightweight LSTM (fewer layers)

---

## Next Steps

- Complete [Mini-Project 5](code/module-5-deep-learning-and-ai-methods.ipynb)
- Review [Module 6: Advanced Topics](module-6-advanced-topics.md) for specialized methods
- Explore [Capstone Project](capstone-project.md) for end-to-end implementation

---

**End of Module 5**
