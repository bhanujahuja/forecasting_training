# Module 0: Introduction & Setup

## 0.1 Overview
- **Objective:** Get started and set up your forecasting environment with Python.

---

### 0.2 What is Forecasting?
- **Definition:** Predicting future outcomes using historical data.
- Forecasting is used in business, science, healthcare, supply chain, and more.

---

### 0.3 Types of Forecasting Covered
- **Time series forecasting:** Predicting variables indexed over time (e.g. sales every month).
- **Non time series forecasting:** Predicting outcomes from general data (classification or regression; e.g. demand prediction for new products).
- **Predictive, Classification & Prescriptive Forecasting:** Various output types and decision-support aims.

---

### 0.4 Installing Python and Packages

1. **Install Python (if not already):**
    - Download & install from [python.org](https://www.python.org/downloads/)
    - Recommended: Python 3.9+

2. **Set up an environment:**  
    - Use [Anaconda](https://www.anaconda.com/products/distribution) (suggested for beginners)  
      OR  
    - Use `virtualenv`:  
      ```bash
      python -m venv forecasting_env
      source forecasting_env/bin/activate  # Linux/Mac
      forecasting_env\Scripts\activate.bat # Windows
      ```

3. **Recommended IDEs:**
   - [Jupyter Notebook/Lab](https://jupyter.org/)
   - [VS Code](https://code.visualstudio.com/)

4. **Install required packages:**  
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn statsmodels prophet xgboost lightgbm tensorflow keras jupyter
    ```

    > *Prophet may require pystan/ephem, [see install notes](https://facebook.github.io/prophet/docs/installation.html).*

---

### 0.5 Testing the Setup

- **Test imports:**
    ```python
    import numpy, pandas, matplotlib, sklearn, statsmodels, prophet, xgboost, tensorflow, keras
    print("All packages imported successfully!")
    ```

- **Start a Jupyter Notebook:**  
    ```bash
    jupyter notebook
    ```
    Open browser → create a new notebook → run the import code above.

---

### 0.6 Course Materials and Datasets

- All course code and notebooks available at:
  [https://github.com/bhanujahuja/forecasting_training](https://github.com/bhanujahuja/forecasting_training)
- Example datasets:  
    - [Airline Passengers](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv)  
    - [Retail Sales](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)  
    - [UCI Machine Learning Repository](https://archive.ics.uci.edu/)  

---

### 0.7 Getting Help

- Official docs for packages ([pandas](https://pandas.pydata.org/docs/), [scikit-learn](https://scikit-learn.org/stable/), etc.)
- Join the course discussion in the GitHub repo!

---

*End of Module 0*