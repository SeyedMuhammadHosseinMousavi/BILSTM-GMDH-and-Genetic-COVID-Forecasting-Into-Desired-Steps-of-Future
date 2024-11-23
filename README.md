# From data to action: Empowering COVID-19 monitoring and forecasting with intelligent algorithms
From data to action: Empowering COVID-19 monitoring and forecasting with intelligent algorithms

# COVID-19 Time Series Forecasting

This repository implements **COVID-19 time series forecasting** using advanced techniques inspired by research from the paper [*From data to action: Empowering COVID-19 monitoring and forecasting through advanced analytics*](https://www.tandfonline.com/doi/full/10.1080/01605682.2023.2240354). It includes methods such as **ARIMA**, **BiLSTM**, **GMDH**, **Genetic Algorithm**, and **TSK Fuzzy Logic** for robust and accurate forecasting of pandemic trends.

![genetic forcast](https://user-images.githubusercontent.com/11339420/147384603-a65c6a99-ed67-4520-86de-ad57aee63ffd.JPG)
---

## Features

- **Forecasting Models**:
  - **ARIMA**: Auto-Regressive Integrated Moving Average for statistical forecasting.
  - **BiLSTM**: Bidirectional Long Short-Term Memory networks for deep learning-based predictions.
  - **GMDH**: Group Method of Data Handling for self-organizing polynomial regression.
  - **Genetic Algorithm**: Evolutionary optimization for parameterized forecasting.
  - **TSK Fuzzy Logic**: Fuzzy inference with Takagi-Sugeno-Kang rules for uncertainty modeling.

- **Preprocessing**:
  - Sliding window feature extraction for time series.
  - Min-Max normalization for feature scaling.

- **Visualization**:
  - Historical vs. forecasted data plots.
  - Comparative performance metrics across models.
![bilstm forecast](https://user-images.githubusercontent.com/11339420/147384602-7ed69c36-c6d0-4f27-aff8-f7a6157190eb.JPG)

---

## 📊 Methodology

### 1**Forecasting Techniques**
- **ARIMA**:
  - A classical statistical model for forecasting time series data.
  - Best suited for univariate data with stationary patterns.

- **BiLSTM**:
  - Deep learning-based model that captures temporal dependencies in both forward and backward directions.
  - Requires large training datasets for optimal performance.

- **GMDH**:
  - A polynomial regression model with self-organizing capabilities.
  - Automatically selects the best subset of features and model complexity.

- **Genetic Algorithm**:
  - Optimizes regression weights by evolving solutions over generations.
  - Suitable for parameter tuning in complex forecasting models.

- **TSK Fuzzy Logic**:
  - Uses Gaussian membership functions for fuzzification.
  - Implements fuzzy `IF-THEN` rules with linear consequents for robust predictions.

---

![gmdh forecast](https://user-images.githubusercontent.com/11339420/147384606-e92fe112-47ce-4c2a-a983-3aa2b4d9a623.JPG)

