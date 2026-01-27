# Module 5: Deep Learning and AI Methods for Forecasting

**Duration:** 8-10 hours (including mini-project)  
**Prerequisites:** Modules 0-4  
**Learning Level:** Advanced

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Build** feedforward, LSTM, and CNN neural networks
2. **Prepare** sequences for deep learning models
3. **Handle** overfitting with dropout and regularization
4. **Train** deep models with callbacks and early stopping
5. **Compare** architectures (LSTM vs CNN vs Hybrid)
6. **Forecast** multiple steps ahead
7. **Deploy** models for production use

---

## Why Deep Learning?

### When Deep Learning Excels

✅ **Excellent for:**
- Very large datasets (10,000+ observations)
- Complex non-linear patterns
- Multi-step forecasting
- Learning hierarchical features automatically
- Transfer learning from pre-trained models

❌ **Not ideal for:**
- Small datasets (< 500 observations)
- Need for interpretability
- Limited computational resources
- When statistical methods work well

---

## 5.1 Sequence Preparation for Neural Networks

### Creating Sequences from Time Series

Neural networks need fixed-size inputs. Convert a series into (X, y) sequences:

```python
import numpy as np

def create_sequences(data, lookback=12, lookahead=1):
    """
    Create sequences for neural network training
    
    data: 1D array
    lookback: number of past steps (history window)
    lookahead: number of steps to predict
    """
    X, y = [], []
    
    for i in range(len(data) - lookback - lookahead + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + lookahead])
    
    return np.array(X), np.array(y)

# Example
series = np.array([10, 12, 14, 13, 15, 18, 20, 22, 21, 23])
X, y = create_sequences(series, lookback=3, lookahead=1)

print("Input sequences (X):")
print(X)
print("\nTargets (y):")
print(y)

# Output:
# X: [[[10, 12, 14],
#      [12, 14, 13],
#      [14, 13, 15], ...]]
# y: [[13], [15], [18], ...]
```

### Scaling for Neural Networks

```python
from sklearn.preprocessing import MinMaxScaler

# Scale to [0, 1] range (important for neural networks)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Create sequences from scaled data
X, y = create_sequences(scaled_data, lookback=12, lookahead=1)

# Split into train, validation, test
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

---

## 5.2 Feedforward Neural Networks

### Simple Dense Network

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Build model
model_ff = keras.Sequential([
    layers.Input(shape=(12,)),  # 12-step history
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output: next value
])

# Compile
model_ff.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train with callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model_ff.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_pred_ff = model_ff.predict(X_test)
mae_ff = np.mean(np.abs(y_test - y_pred_ff))
print(f"Feedforward MAE: {mae_ff:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Training History - Feedforward')
plt.grid()
plt.show()
```

---

## 5.3 LSTM: Recurrent Neural Networks

### LSTM Architecture

```python
# LSTM model for sequence forecasting
model_lstm = keras.Sequential([
    layers.Input(shape=(12, 1)),  # (timesteps, features)
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu', return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Important: Reshape X for LSTM (samples, timesteps, features)
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train
history_lstm = model_lstm.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Predict and evaluate
y_pred_lstm = model_lstm.predict(X_test_lstm)
mae_lstm = np.mean(np.abs(y_test - y_pred_lstm))
print(f"LSTM MAE: {mae_lstm:.4f}")
```

### Bidirectional LSTM

```python
# LSTM that reads forward AND backward
model_bilstm = keras.Sequential([
    layers.Input(shape=(12, 1)),
    layers.Bidirectional(
        layers.LSTM(64, activation='relu', return_sequences=True)
    ),
    layers.Dropout(0.2),
    layers.Bidirectional(
        layers.LSTM(32, activation='relu')
    ),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_bilstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

history_bilstm = model_bilstm.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

y_pred_bilstm = model_bilstm.predict(X_test_lstm)
mae_bilstm = np.mean(np.abs(y_test - y_pred_bilstm))
print(f"Bidirectional LSTM MAE: {mae_bilstm:.4f}")
```

---

## 5.4 1D Convolutional Neural Networks

### CNN for Time Series

```python
# 1D CNN: Local pattern detection
model_cnn = keras.Sequential([
    layers.Input(shape=(12, 1)),
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Dropout(0.2),
    layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),
    layers.Conv1D(16, kernel_size=3, activation='relu'),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

history_cnn = model_cnn.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

y_pred_cnn = model_cnn.predict(X_test_lstm)
mae_cnn = np.mean(np.abs(y_test - y_pred_cnn))
print(f"CNN MAE: {mae_cnn:.4f}")
```

**Why CNN works for time series:**
- Fast inference
- Efficient parameter usage
- Good for local temporal patterns
- Can use large history windows

---

## 5.5 Hybrid Architectures

### CNN-LSTM Combination

```python
# Combine CNN's pattern detection with LSTM's sequence learning
model_hybrid = keras.Sequential([
    layers.Input(shape=(12, 1)),
    # CNN layers
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Dropout(0.2),
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.Dropout(0.2),
    # LSTM layers
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(16, activation='relu'),
    # Dense layers
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_hybrid.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

history_hybrid = model_hybrid.fit(
    X_train_lstm, y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

y_pred_hybrid = model_hybrid.predict(X_test_lstm)
mae_hybrid = np.mean(np.abs(y_test - y_pred_hybrid))
print(f"CNN-LSTM MAE: {mae_hybrid:.4f}")
```

---

## 5.6 Multi-Step Ahead Forecasting

### Direct Approach: Train for Multiple Steps

```python
# Create sequences that predict 3 steps ahead
X_multi, y_multi = create_sequences(scaled_data, lookback=12, lookahead=3)

# Split
train_size = int(len(X_multi) * 0.7)
X_train_m = X_multi[:train_size].reshape(train_size, 12, 1)
y_train_m = y_multi[:train_size]
X_test_m = X_multi[train_size:].reshape(len(X_multi) - train_size, 12, 1)
y_test_m = y_multi[train_size:]

# Model
model_multi = keras.Sequential([
    layers.Input(shape=(12, 1)),
    layers.LSTM(64, activation='relu', return_sequences=True),
    layers.LSTM(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3)  # 3-step ahead
])

model_multi.compile(optimizer='adam', loss='mse')
model_multi.fit(X_train_m, y_train_m, epochs=50, batch_size=32, verbose=0)

# Predict 3 steps ahead
y_pred_multi = model_multi.predict(X_test_m)
print(f"Multi-step predictions shape: {y_pred_multi.shape}")
```

### Recursive Approach: One Step at a Time

```python
def forecast_recursive(model, initial_sequence, n_steps):
    """
    Forecast n steps ahead by repeatedly predicting next step
    """
    predictions = []
    current_seq = initial_sequence.copy()
    
    for _ in range(n_steps):
        # Predict next step
        next_pred = model.predict(current_seq.reshape(1, 12, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence (remove first, add prediction)
        current_seq = np.append(current_seq[1:], next_pred)
    
    return np.array(predictions)

# Example: Forecast 12 steps ahead
initial_seq = X_test_lstm[0].flatten()
future_forecast = forecast_recursive(model_lstm, initial_seq, n_steps=12)
print(f"12-step recursive forecast: {future_forecast}")
```

---

## 5.7 Model Comparison

```python
# Compare all architectures
architectures = {
    'Feedforward': y_pred_ff,
    'LSTM': y_pred_lstm,
    'BiLSTM': y_pred_bilstm,
    'CNN': y_pred_cnn,
    'CNN-LSTM': y_pred_hybrid
}

results = {}
for name, preds in architectures.items():
    mae = np.mean(np.abs(y_test - preds))
    rmse = np.sqrt(np.mean((y_test - preds) ** 2))
    results[name] = {'MAE': mae, 'RMSE': rmse}

comparison_df = pd.DataFrame(results).T.sort_values('MAE')
print("\nDeep Learning Models Comparison:")
print(comparison_df)

# Visualize
plt.figure(figsize=(14, 6))
plt.plot(y_test[:50], label='Actual', marker='o', linewidth=2)
for name, preds in architectures.items():
    plt.plot(preds[:50], label=name, alpha=0.7)
plt.legend()
plt.title('Deep Learning Architectures Comparison')
plt.grid(alpha=0.3)
plt.show()
```

---

## 5.8 Mini Project: End-to-End Deep Learning

```python
# Complete pipeline
# 1. Load and scale
df = pd.read_csv(...)
series = df['value'].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# 2. Create sequences
X, y = create_sequences(scaled, lookback=12)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. Split
train_idx = int(len(X) * 0.7)
val_idx = int(len(X) * 0.85)
X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
X_test, y_test = X[val_idx:], y[val_idx:]

# 4. Build best model (LSTM)
model = keras.Sequential([
    layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(12, 1)),
    layers.Dropout(0.2),
    layers.LSTM(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_loss', patience=10),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ],
    verbose=0
)

# 6. Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {np.mean(np.abs(y_test - y_pred)):.4f}")

# 7. Forecast future
future_seq = X_test[-1].copy().flatten()
future_forecast = forecast_recursive(model, future_seq, n_steps=12)

# 8. Inverse scale
future_forecast_original = scaler.inverse_transform(future_forecast.reshape(-1, 1))
print(f"Future forecast: {future_forecast_original.flatten()}")
```

---

## 5.9 Best Practices

✅ **Do This:**
- Scale data to [0, 1] or standardize
- Use dropout and early stopping to prevent overfitting
- Start with simple architecture, increase complexity if needed
- Use time-series splits (not random)
- Validate on completely separate test set
- Try multiple architectures and compare

❌ **Don't Do This:**
- Forget to reshape data correctly (need 3D for RNN/CNN)
- Use too many parameters (overfitting)
- Mix train/test data during scaling
- Randomly shuffle sequences
- Skip validation set

---

## Progress Checkpoint

**Completion: ~65%** ✓

You now understand:
- Sequence preparation for deep learning
- Feedforward, LSTM, BiLSTM architectures
- 1D CNNs for efficient pattern detection
- Hybrid CNN-LSTM models
- Multi-step ahead forecasting
- Model comparison and selection

**Next Module (Module 6):** Advanced Topics
- Multivariate forecasting
- Anomaly detection
- Ensemble methods across all paradigms
- Production deployment

**Time Estimate for Module 5:** 8-10 hours  
**Estimated Combined Progress:** 65% → 80%

---

*End of Module 5: Deep Learning and AI Methods for Forecasting*
