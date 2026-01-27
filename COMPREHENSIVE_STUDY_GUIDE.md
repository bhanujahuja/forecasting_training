# Comprehensive Study Guide for the Forecasting Course

**How to Use This Guide:**
This document provides:
- Clear learning paths for different levels
- Study checklists for each module
- Common misconceptions clarified
- Worked examples and detailed walkthroughs
- Practice problems with solutions
- Tips for success

---

## Table of Contents
1. [How to Use This Course](#how-to-use-this-course)
2. [Quick Reference: When to Use Each Method](#quick-reference)
3. [Detailed Learning Pathways](#learning-pathways)
4. [Module-by-Module Study Tips](#module-study-tips)
5. [Common Misconceptions](#misconceptions)
6. [Practice Problems & Solutions](#practice-problems)
7. [Glossary of Key Terms](#glossary)

---

## How to Use This Course

### For Self-Learners (Recommended Approach)

**Weekly Schedule (5-8 weeks total):**
```
Week 1-2:  Modules 0-1 (Setup, Fundamentals)
Week 2-3:  Module 2 (Basic Methods)
Week 3-4:  Module 3 (Statistical Methods)
Week 4-5:  Module 4 (Machine Learning)
Week 5-6:  Module 5 (Deep Learning)
Week 6-7:  Module 6 (Advanced Topics)
Week 7-8:  Capstone Project
```

**Daily Commitment:**
- 1 hour reading/theory
- 1-2 hours hands-on coding
- 30 minutes review/notes
- **Total: 2.5-3.5 hours per day**

### For Structured Learners

**Follow this order strictly:**
1. Read markdown file completely
2. Run notebook cells one at a time
3. Modify code and experiment
4. Complete mini-project
5. Answer knowledge check questions
6. Only then proceed to next module

### For Visual Learners

**Focus on:**
- All plots and visualizations
- Code output and results
- Comparison tables
- Flowcharts in this guide

---

## Quick Reference: When to Use Each Method

```
┌─ Do you have time series data?
│  │
│  ├─ YES: Continue below
│  │       │
│  │       ├─ Is the trend consistent?
│  │       │  ├─ NO → Use ARIMA/SARIMA (Module 3)
│  │       │  └─ YES → Is there seasonality?
│  │       │           ├─ NO → Linear Regression (Module 2) or ARIMA
│  │       │           └─ YES → Holt-Winters (Module 2) or SARIMA (Module 3)
│  │       │
│  │       ├─ Is data highly non-linear?
│  │       │  ├─ NO → Try Statistical Methods (Module 3) first
│  │       │  └─ YES → Use ML (Module 4) or DL (Module 5)
│  │       │
│  │       └─ Do you have exogenous variables?
│  │          ├─ NO → Use univariate methods above
│  │          └─ YES → Use ARIMAX (Module 3) or ML with features (Module 4)
│  │
│  └─ NO: Use cross-sectional ML (Module 4)
│          (Or classical regression)
```

---

## Learning Pathways

### Beginner (0-2 years of Python/ML experience)

**Week 1-2: Foundation**
- [ ] Module 0: Setup (follow all steps carefully)
- [ ] Module 1: Fundamentals (complete all visualizations)
- [ ] **First Mini-Project:** EDA on chosen dataset
- [ ] Review: Can you identify trends and seasonality?

**Week 2-3: Simple Methods**
- [ ] Module 2: Basic Methods (implement each method)
- [ ] **Second Mini-Project:** Compare 5+ baseline methods
- [ ] Review: Can you choose best baseline for your data?

**Week 4: Statistical Methods**
- [ ] Module 3: Statistical Methods (focus on ARIMA/SARIMA)
- [ ] **Third Mini-Project:** Build ARIMA model with diagnostics
- [ ] Review: Can you interpret ACF/PACF plots?

**Week 5: ML and Beyond**
- [ ] Module 4: Machine Learning (feature engineering focus)
- [ ] Module 5: Deep Learning (LSTM focus)
- [ ] **Fourth Mini-Project:** Compare ML/DL with statistical methods

**Week 5-6: Integration**
- [ ] Module 6: Advanced Topics (skim for awareness)
- [ ] Capstone Project: Full end-to-end project
- [ ] **Final Deliverable:** Complete capstone with all phases

### Intermediate (2-5 years of Python/ML experience)

**Week 1: Fast Track**
- [ ] Module 0: Quick review of setup
- [ ] Module 1: Review fundamentals (15 minutes)
- [ ] Module 2: Review baseline methods (30 minutes)
- [ ] **Action:** Build baseline models on your data

**Week 1-2: Core Methods**
- [ ] Module 3: Detailed study of ARIMA/SARIMA
- [ ] Module 4: Feature engineering deep dive
- [ ] **Action:** Build statistical + ML pipeline

**Week 2-3: Advanced**
- [ ] Module 5: All neural architectures
- [ ] Module 6: Advanced topics matching your interests
- [ ] **Action:** Build ensemble of 5+ models

**Week 3-4: Integration**
- [ ] Capstone Project: Full execution with research
- [ ] **Final Deliverable:** Production-ready code

### Advanced (5+ years of Python/ML/Statistics)

**Week 1: Assessment**
- [ ] Modules 0-3: Skim for familiarization
- [ ] Identify gaps in knowledge
- [ ] **Action:** Build baseline statistical models

**Week 1-2: Specialization**
- [ ] Module 4-6: Deep dive into methods matching your interests
- [ ] **Action:** Extend methods with custom techniques

**Week 2-3: Innovation**
- [ ] Capstone Project: Research extension
- [ ] Implement novel approaches
- [ ] **Final Deliverable:** Research-grade analysis

---

## Module-by-Module Study Tips

### Module 0: Setup (2-3 hours)

**Success Checklist:**
- [ ] Python installed and verified
- [ ] All packages install without errors
- [ ] Jupyter notebook launches
- [ ] Test script runs successfully
- [ ] You can run a simple pandas command
- [ ] You understand what each package does

**Common Issues:**
- **Prophet installation fails**: `pip install pystan==2.19.1.1` first
- **Jupyter won't start**: Check virtual environment is activated
- **Import errors**: Make sure `pip install` completed successfully

**Pro Tips:**
- Keep a terminal open to reference installation
- Bookmark official docs for quick lookup
- Start with Jupyter - it's most beginner-friendly

---

### Module 1: Fundamentals (4-5 hours)

**Success Checklist:**
- [ ] Can explain difference between time series and cross-sectional data
- [ ] Can load and inspect a dataset (shape, dtypes, nulls)
- [ ] Can identify trends in a plot (visual inspection)
- [ ] Can identify seasonality and its period
- [ ] Can explain why you can't shuffle time series
- [ ] Can create train/val/test splits respecting time order
- [ ] Have completed the mini-project with 5+ visualizations

**Key Concept - Why This Matters:**
Time series = **temporal order matters**
Cross-sectional = **order doesn't matter**

This is THE fundamental difference that affects everything else.

**Must Create Visualizations:**
1. Raw time series line plot
2. Distribution/histogram
3. Rolling mean/trend
4. Seasonal subseries plot
5. Autocorrelation plot
6. Decomposition (if seasonal)

**Practice Problem:**
"I have daily stock prices for 2 years. Is this time series or cross-sectional? Why? What patterns might I see?"
- Answer: Time series. Order matters (past prices influence future). Look for trend, maybe daily/weekly seasonality, but likely random walk behavior.

---

### Module 2: Basic Methods (6-8 hours)

**Success Checklist:**
- [ ] Implemented all 7 baseline methods
- [ ] Understand when to use each
- [ ] Can calculate MAE, RMSE, MAPE by hand
- [ ] Can create train/test split correctly
- [ ] Know why seasonal naive often beats complex models
- [ ] Have created comparison table (methods × metrics)
- [ ] Have completed mini-project with analysis

**The Baseline Truth:**
"Don't build a fancy model without knowing how the simple ones perform."

**Key Methods to Master:**
1. **Mean Forecast**: Always use as absolute baseline
2. **Naive**: Works surprisingly well for random walks
3. **Seasonal Naive**: Often the winner for seasonal data
4. **Moving Average**: Quick trend identifier
5. **Exponential Smoothing**: Adaptive smoothing
6. **Holt-Winters**: Trend + Seasonality powerhouse
7. **Linear Regression**: Captures linear trend

**Common Mistake:**
Thinking Holt-Winters is "basic" - it's actually very powerful! Many practitioners just use H-W without going further.

**Must Create Visualizations:**
1. All forecasts vs actual on same plot
2. Error comparison (bar chart)
3. Error distribution over time (line plot)
4. Residuals plot
5. Metrics comparison table

**Practice Problem:**
"You built 6 baseline models. MAE: [2.1, 2.2, 1.8, 3.2, 1.9, 2.0]. Which do you choose? Why?"
- Answer: 1.8 (lowest MAE), but check RMSE and MAPE too. If similar, choose simplest. Check if 1.8 is just lucky on test set (cross-validate).

---

### Module 3: Statistical Methods (8-10 hours)

**Success Checklist:**
- [ ] Can read and interpret ACF/PACF plots
- [ ] Understand stationarity and can test for it (ADF test)
- [ ] Can difference data and check for white noise residuals
- [ ] Can build ARIMA(p,d,q) models
- [ ] Can build SARIMA(p,d,q,P,D,Q,s) models
- [ ] Understand auto.arima() and parameter selection
- [ ] Can interpret diagnostic plots
- [ ] Have completed mini-project with proper validation

**The ARIMA Advantage:**
ARIMA can capture complex patterns with few parameters. If data is stationary or can be differenced, ARIMA is hard to beat.

**Critical Concept - Stationarity:**
```
Non-stationary (trend):    Stationary (after differencing):
┌───────┐                  ┌───────┐
│   ╱   │                  │ ┌─┐┌─┐│
│  ╱    │                  │ └─┘└─┘│
│ ╱     │                  │ ┌─┐┌─┐│
└───────┘                  └─────────┘

Mean changes with time     Mean is constant
Variance changes           Variance is constant
ACF decays slowly           ACF decays quickly
```

**Key Parameters Explained:**
- **p**: Number of autoregressive terms (past values)
- **d**: Number of times to difference (make stationary)
- **q**: Number of moving average terms (past errors)
- **P, D, Q, s**: Seasonal versions of p, d, q, period

**Must Create Visualizations:**
1. Original + differenced series
2. ACF/PACF of differenced series
3. Time series decomposition
4. Diagnostic plots (4-panel from ARIMA)
5. Forecast with confidence intervals

**Practice Problem:**
"ACF decays slowly, PACF cuts off at lag 1. What ARIMA(p,d,q) would you try?"
- Answer: Likely need differencing (d=1), then ARIMA(1,1,0) or try auto.arima()

---

### Module 4: Machine Learning (10-12 hours)

**Success Checklist:**
- [ ] Can engineer 10+ feature types from time series
- [ ] Understand lag features, rolling statistics, trend, seasonality
- [ ] Can build Linear Regression, Random Forest, XGBoost, LightGBM
- [ ] Know time-series cross-validation (TimeSeriesSplit)
- [ ] Can hyperparameter tune with GridSearchCV
- [ ] Understand feature importance
- [ ] Can compare ML vs statistical methods
- [ ] Have completed mini-project with 5+ models

**Feature Engineering is 80% of ML Success:**
Good features beat fancy algorithms. Spend time here!

**Feature Ideas (Steal These!):**
```
1. Lags: y[t-1], y[t-7], y[t-30], y[t-365]
2. Rolling: mean, std, min, max (windows: 7, 30, 90)
3. Trend: Linear trend, polynomial trend
4. Seasonality: sin/cos of day/month/year
5. Exogenous: Weather, holidays, campaigns
6. Interactions: lag1 * month, trend * day_of_week
7. Decomposition: Trend, seasonal, residual components
8. Temporal: day_of_week, is_weekend, is_holiday
9. Statistical: autocorrelation, entropy, complexity
```

**ML Advantages Over ARIMA:**
- Handle non-linear relationships
- Use exogenous variables easily
- Easier with multiple series
- Can capture complex interactions

**Disadvantages:**
- Need more data
- Black box (harder to interpret)
- Can overfit if not careful
- Slower to train/predict

**Must Create Visualizations:**
1. Feature importance comparison
2. Actual vs predicted (scatter + line)
3. Residuals plot
4. Error distribution
5. Learning curves (train vs test error)
6. Cross-validation scores

**Practice Problem:**
"You have 100 features. How do you avoid overfitting?"
- Answer: Time-series cross-validation, regularization (L1/L2), feature selection, test on held-out future data, simple models first.

---

### Module 5: Deep Learning (10-12 hours)

**Success Checklist:**
- [ ] Understand sequence preparation (lookback window)
- [ ] Can build Feedforward, LSTM, CNN, CNN-LSTM models
- [ ] Know proper data normalization (MinMaxScaler)
- [ ] Can train with callbacks (EarlyStopping, ReduceLROnPlateau)
- [ ] Understand underfitting vs overfitting
- [ ] Can create ensembles (averaging, voting)
- [ ] Can evaluate with proper metrics
- [ ] Have completed mini-project with 4+ architectures

**When DL Wins:**
- Very long sequences (100+ timesteps)
- Complex non-linear relationships
- Large datasets (1000+ samples)
- Multiple related series
- Multi-step ahead forecasting

**When DL Loses:**
- Small datasets (<500 samples)
- Data is highly seasonal (statistical methods win)
- Interpretability is critical
- Need uncertainty estimates
- Computational resources limited

**Sequence Preparation (Critical!):**
```python
# Wrong: Feed all data at once
X = entire_dataset  # Shape: (1000,)

# Right: Create sequences
X = [
    [1, 2, 3, 4, 5],      # t=0 to t=4 predicts t=5
    [2, 3, 4, 5, 6],      # t=1 to t=5 predicts t=6
    [3, 4, 5, 6, 7],      # t=2 to t=6 predicts t=7
    ...
]  # Shape: (N, lookback, 1)
```

**Architecture Choices:**
- **Feedforward**: Simple baseline, fast
- **LSTM**: Best for sequence dependency, memory of long-term patterns
- **CNN**: Good for local patterns, fast
- **CNN-LSTM**: Best of both worlds
- **Ensemble**: Average all 4 models

**Must Create Visualizations:**
1. Training history (loss, MAE curves)
2. Predictions vs actual
3. Residuals and errors
4. Feature maps (for CNN)
5. Hidden state visualization (for LSTM)
6. Ensemble comparison

**Practice Problem:**
"Your LSTM overfits (train loss << val loss). How do you fix it?"
- Answer: Add dropout, reduce model size, regularization (L1/L2), more data, early stopping, reduce training epochs.

---

### Module 6: Advanced Topics (5-7 hours)

**Success Checklist:**
- [ ] Understand anomaly detection methods
- [ ] Know when to use multivariate forecasting
- [ ] Can identify change points
- [ ] Understand prescriptive analytics
- [ ] Know basics of reinforcement learning
- [ ] Have awareness of production considerations
- [ ] Have skimmed all section and pick 2 deep dives

**Study Strategy:**
Module 6 is reference material. Pick 2 topics most relevant to your goals:
- **Topic A: Anomalies** - If data has unusual events
- **Topic B: Change Points** - If data has structural breaks
- **Topic C: Multivariate** - If forecasting multiple series
- **Topic D: Prescriptive** - If need to optimize decisions
- **Topic E: RL** - If interest in adaptive systems

**No Mini-Project:** Module 6 is theory. Deep dives in Capstone.

---

### Capstone Project (8-12 hours)

**Success Checklist:**
- [ ] Have chosen dataset and defined problem
- [ ] Completed EDA with insights
- [ ] Built 5+ models (baseline + advanced)
- [ ] Created ensemble with clear logic
- [ ] Written technical report (500-1000 words)
- [ ] Made business recommendations
- [ ] Code is clean and commented
- [ ] Delivered all 8 phases

**Capstone Success Factors:**
1. **Choose good data**: Clear pattern, relevant to you
2. **Document everything**: How did you get 95% accuracy?
3. **Show comparison**: Model A vs Model B vs Ensemble
4. **Interpret results**: Why does XGBoost beat LSTM?
5. **Business framing**: What decisions does forecast enable?
6. **Honest assessment**: What limitations does model have?
7. **Future work**: How would you improve?

**Capstone Deliverables (Minimum):**
- [ ] Jupyter notebook (all 8 phases)
- [ ] Technical report (PDF or Markdown)
- [ ] Model comparison table
- [ ] 3+ visualizations showing results
- [ ] README explaining how to run code

**Capstone Deliverables (Professional Quality):**
- [ ] All above, plus:
- [ ] Trained model saved (.pkl, .h5)
- [ ] Production code (inference script)
- [ ] Requirements.txt or environment.yml
- [ ] CI/CD ready (tests pass)
- [ ] Deployment instructions

---

## Common Misconceptions

### 1. "More Complex = Better"
**❌ Wrong:** A PhD thesis model vs. Excel Holt-Winters
**✅ Right:** Seasonal naive often beats complex models

**Why?** Over-parameterization, overfitting, data limitations

**Lesson:** Always establish baselines first. Choose simplest model that works.

---

### 2. "I Should Use All My Data for Training"
**❌ Wrong:** Train on 100%, evaluate on... nothing
**✅ Right:** 60% train, 20% val, 20% test

**Why?** You need honest evaluation on unseen data

**Lesson:** Split data FIRST, before any modeling

---

### 3. "Shuffle Your Data for Better ML"
**❌ Wrong (for time series):** Mix past with future
**✅ Right:** Respect temporal order

**Why?** Time series has dependencies; shuffling creates information leakage

**Lesson:** Use TimeSeriesSplit, not random split

---

### 4. "Lower MAE Always Means Better Model"
**❌ Wrong:** Model A: MAE=10, Model B: MAE=11 → Use A
**✅ Right:** Consider RMSE, MAPE, interpretability, speed, robustness

**Why?** Different metrics reveal different things

**Lesson:** Use multiple metrics. Context matters.

---

### 5. "Deep Learning Always Beats Classical Methods"
**❌ Wrong:** Build an LSTM because it's trendy
**✅ Right:** Try ARIMA first, then ML, then DL

**Why?** DL needs more data, more tuning, more compute

**Lesson:** Start simple, increase complexity only if needed

---

### 6. "Stationarity Doesn't Matter"
**❌ Wrong:** Feed non-stationary data to ARIMA
**✅ Right:** Check ADF test, difference if needed

**Why?** ARIMA assumes stationarity; violating this breaks assumptions

**Lesson:** Always test for stationarity before ARIMA

---

### 7. "One Train/Test Split is Enough"
**❌ Wrong:** Evaluate on one random split
**✅ Right:** Use cross-validation with multiple splits

**Why?** One split might be lucky. Need robust estimates

**Lesson:** Use TimeSeriesSplit for time series cross-validation

---

### 8. "My Training Error is 2%, My Test Error is 50%"
**❌ Panic Mode:** "Model is broken!"
**✅ Investigate:** Classic overfitting. Normal.

**Why?** Model memorized training patterns not present in test

**Lesson:** Regularization, simpler model, more data, or dropout

---

## Practice Problems & Solutions

### Problem Set 1: Data Understanding

**Problem 1.1:** You have hourly temperature data for 1 year (8,760 observations). Is this time series or cross-sectional?
- **Answer:** Time series. Order matters - temperature today affects tomorrow. Can't shuffle.

**Problem 1.2:** ACF plot shows spikes at lags 1, 2, 3, ... decaying slowly. What does this mean?
- **Answer:** Non-stationary data. Probably has a trend. Try differencing.

**Problem 1.3:** Your seasonal naive model has MAE=5.2, your linear regression has MAE=6.1. Which is better?
- **Answer:** Seasonal naive (lower MAE). But check RMSE and MAPE too. Also cross-validate to ensure not lucky.

---

### Problem Set 2: Method Selection

**Problem 2.1:** Data has clear yearly seasonality, growing trend, and monthly granularity. What method should you try first?
- **Answer:** Holt-Winters (additive or multiplicative) or SARIMA
- **Why:** Both handle trend + seasonality natively
- **Second try:** If HW works, done! If not, try SARIMA

**Problem 2.2:** You have stock prices (daily for 5 years). What's your strategy?
- **Answer:** 
  1. Seasonal naive (if any day-of-week pattern)
  2. Try ARIMA (likely random walk)
  3. Try ML with features (volume, momentum, etc.)
  4. Try ensemble of all

**Problem 2.3:** Demand for ice cream (seasonal) plus promotions (event variable). What method?
- **Answer:** Machine Learning with features:
  - Lag features (yesterday's sales)
  - Seasonal features (day of year)
  - Event indicator (was there a promo?)
  - External: temperature if available
- **Why:** Easy to add exogenous variables in ML; harder in ARIMA

---

### Problem Set 3: Implementation

**Problem 3.1:** Write pseudo-code to implement seasonal naive forecast:
```python
# Pseudo-code
def seasonal_naive_forecast(history, seasonality_period):
    # Your history is months 1-36
    # You want to forecast months 37-48
    
    forecasts = []
    for forecast_month in range(37, 49):
        # What's the value from 12 months ago?
        past_month = forecast_month - seasonality_period  # 25, 26, ...
        forecasts.append(history[past_month])
    return forecasts
```

**Solution:**
```python
def seasonal_naive_forecast(history, seasonality_period):
    forecasts = []
    for i in range(seasonality_period):  # Forecast next season
        past_value = history[-(seasonality_period - i)]  # Value from last season
        forecasts.append(past_value)
    return forecasts
```

**Problem 3.2:** You want to build an ARIMA model. Describe the steps:
```
1. Test for stationarity (ADF test)
2. If non-stationary, difference the data
3. Plot ACF/PACF to identify p and q
4. Try auto_arima() to find best ARIMA(p,d,q)
5. Fit model on training data
6. Make forecasts on test data
7. Check diagnostic plots
8. Evaluate on test set
```

**Problem 3.3:** Explain why you can't use random_state for shuffling time series:
```
WRONG:
indices = [0, 5, 2, 8, 1, ...]  # Random permutation
train = df.iloc[indices[:80%]]  # Training and test are mixed
test = df.iloc[indices[80%:]]   # Test contains past info!
Result: Model sees future data, overly optimistic accuracy

RIGHT:
train = df[:80%]  # Training gets first 80% in order
test = df[80%:]   # Test gets last 20% in order
Result: Model only predicts future from past. Realistic.
```

---

### Problem Set 4: Debugging

**Problem 4.1:** Your Holt-Winters forecast looks flat (horizontal line). Why?
- **Possible causes:**
  1. Smoothing parameters (α, β, γ) are near 0 (not adaptive)
  2. Data is too noisy
  3. Model didn't fit well
- **Solutions:**
  1. Check fitted model parameters
  2. Try with optimized=True to auto-tune
  3. Pre-smooth data with moving average
  4. Add seasonal_periods parameter

**Problem 4.2:** ARIMA(1,1,1) forecast diverges (goes to infinity). What's wrong?
- **Likely cause:** Model is unstable (unit root)
- **Solutions:**
  1. Increase differencing (d=2)
  2. Try different (p,q) values
  3. Use auto.arima()
  4. Check for structural breaks (level shifts)

**Problem 4.3:** LSTM has training loss=100, validation loss=500. Massive overfitting. Fix it:
- **Solutions (in order):**
  1. Add dropout (0.2-0.5)
  2. Add L1/L2 regularization
  3. Reduce model size (fewer LSTM units)
  4. Add more training data
  5. Use early stopping
  6. Reduce training epochs

---

## Glossary of Key Terms

### Time Series Specific

**Autocorrelation (ACF):** Correlation of a series with its past values. High ACF = past strongly predicts future.

**Stationarity:** Statistical properties (mean, variance) don't change over time. Required for ARIMA.

**Differencing:** Taking differences y[t] - y[t-1] to make non-stationary data stationary.

**Seasonality:** Regular, repeating patterns at fixed intervals (daily, weekly, yearly).

**Trend:** Overall direction (up or down) in data over time.

**Lag:** Past value at a specified time distance. Lag-1 is immediately previous value.

### Model-Specific

**ARIMA(p,d,q):** Autoregressive Integrated Moving Average. p=AR, d=Differencing, q=MA.

**SARIMA:** Seasonal ARIMA. Adds seasonal components (P,D,Q,s).

**Exponential Smoothing:** Weighting scheme that gives more importance to recent observations.

**Cross-validation:** Technique to evaluate model on multiple train/test splits.

**Overfitting:** Model memorizes training data; poor test performance.

**Underfitting:** Model too simple; poor training and test performance.

### Metric-Specific

**MAE (Mean Absolute Error):** Average absolute difference. Interpretable, same units as data.

**RMSE (Root Mean Squared Error):** Penalizes large errors. Sensitive to outliers.

**MAPE (Mean Absolute Percentage Error):** Percentage error. Good for comparing different scales.

### Deep Learning

**LSTM (Long Short-Term Memory):** RNN variant that remembers long-term dependencies.

**Dropout:** Regularization technique that randomly disables neurons during training.

**Callback:** Function that runs during training (e.g., EarlyStopping).

**Sequence:** Ordered list of values used to predict next value.

---

## Additional Resources

### Official Documentation
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/tsa.html)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [TensorFlow LSTM](https://www.tensorflow.org/guide/keras/rnn)

### Recommended Reading
- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos (free online)
- "Time Series Analysis by State Space Methods" by Durbin & Koopman
- Kaggle competition: [Store Sales Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting)

### Online Communities
- Stack Overflow (tag: time-series, arima, forecasting)
- Cross Validated (Statistics Q&A)
- Kaggle Competitions (real datasets, solutions)

---

**Last Updated:** January 28, 2026  
**Maintained By:** Course Team  
**Version:** 1.0

*This guide is meant to be a living document. Share feedback and suggestions!*
