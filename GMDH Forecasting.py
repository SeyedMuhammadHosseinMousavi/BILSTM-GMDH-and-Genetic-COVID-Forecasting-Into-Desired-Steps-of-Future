%reset -f

# Group Method of Data Handling(GMDH)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

# Load the dataset
file_path = 'time_series_covid19_confirmed_global.csv'
covid_data = pd.read_csv(file_path)

# Select the country (Iran in this case)
country_name = 'Iran'
country_data = covid_data[covid_data['Country/Region'] == country_name]

# Aggregate data for the country (if multiple provinces exist)
country_data = country_data.iloc[:, 4:].sum()  # Skip Lat, Long, and metadata
country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y')
data = country_data.values.reshape(-1, 1)  # Reshape to a 2D array for GMDH

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data for GMDH
look_back = 20  # Number of previous days to use for prediction
X, y = [], []
for i in range(len(data_normalized) - look_back):
    X.append(data_normalized[i:i + look_back, 0])
    y.append(data_normalized[i + look_back, 0])
X, y = np.array(X), np.array(y)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define parameters for GMDH
max_neurons_per_layer = 9  # Maximum number of neurons (features) in each layer
max_layers = 10  # Maximum number of layers
polynomial_degree = 3  # Degree of polynomial combinations (e.g., quadratic)

# Define a function to create polynomial combinations
def generate_features(X, degree=2):
    n_features = X.shape[1]
    feature_combinations = []
    for i, j in combinations(range(n_features), 2):
        if degree >= 2:
            combined_feature = X[:, i] * X[:, j]
            combined_feature = np.clip(combined_feature, -1e6, 1e6)  # Clip large values to avoid overflow
            feature_combinations.append(combined_feature)
    return np.hstack((X, np.array(feature_combinations).T))

# Train the GMDH-like model
best_model = None
min_error = float('inf')

for layer in range(max_layers):  # Repeat the process for multiple layers
    # Generate polynomial features
    X_train_poly = generate_features(X_train, degree=polynomial_degree)
    X_test_poly = generate_features(X_test, degree=polynomial_degree)

    # Refit the MinMaxScaler for each new set of features
    poly_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_poly = poly_scaler.fit_transform(X_train_poly)
    X_test_poly = poly_scaler.transform(X_test_poly)

    # Fit a simple linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    # Calculate error
    error = np.mean((y_pred - y_test) ** 2)
    if error < min_error:
        min_error = error
        best_model = model

    # Select the top features based on importance or performance
    # Limit the number of neurons to max_neurons_per_layer
    X_train_poly_sorted = np.argsort(np.abs(model.coef_))[-max_neurons_per_layer:]
    X_test_poly_sorted = np.argsort(np.abs(model.coef_))[-max_neurons_per_layer:]

    X_train = X_train_poly[:, X_train_poly_sorted]
    X_test = X_test_poly[:, X_test_poly_sorted]

# Forecast future values
forecast_days = 200
forecast_input = data_normalized[-look_back:].reshape(1, -1)
forecast_output = []

for _ in range(forecast_days):
    # Generate polynomial features
    forecast_input_poly = generate_features(forecast_input, degree=polynomial_degree)
    
    # Normalize polynomial features for the current iteration
    poly_scaler = MinMaxScaler(feature_range=(0, 1))
    forecast_input_poly = poly_scaler.fit_transform(forecast_input_poly)
    
    next_value = best_model.predict(forecast_input_poly)[0]
    forecast_output.append(next_value)

    # Update the input with the new prediction
    forecast_input = np.append(forecast_input[:, 1:], [[next_value]], axis=1)

# Transform forecast back to original scale
forecast_output = scaler.inverse_transform(np.array(forecast_output).reshape(-1, 1))

# Create a timeline for the forecast
forecast_index = pd.date_range(start=country_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

# Plot historical and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(country_data.index, country_data.values, label='Historical Data', marker='o')
plt.plot(forecast_index, forecast_output, label='Forecasted Data', linestyle='--', marker='o', color='red')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title(f'COVID-19 Forecast for {country_name} using Custom GMDH-like Algorithm')
plt.legend()
plt.grid()
plt.show()
