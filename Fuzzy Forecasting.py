
%reset -f

# Fuzzy (TSK)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'time_series_covid19_confirmed_global.csv'
covid_data = pd.read_csv(file_path)

# Select the country (Iran in this case)
country_name = 'Iran'
country_data = covid_data[covid_data['Country/Region'] == country_name]

# Aggregate data for the country (if multiple provinces exist)
country_data = country_data.iloc[:, 4:].sum()  # Skip Lat, Long, and metadata
country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y')
data = country_data.values.reshape(-1, 1)  # Reshape to a 2D array for TSK

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data for TSK fuzzy logic
look_back = 15  # Number of previous days to use for prediction
X, y = [], []
for i in range(len(data_normalized) - look_back):
    X.append(data_normalized[i:i + look_back, 0])
    y.append(data_normalized[i + look_back, 0])
X, y = np.array(X), np.array(y)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define fuzzy membership functions (Gaussian)
def gaussian_mf(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)

# Initialize fuzzy parameters
num_rules = 5  # Number of fuzzy rules
centers = np.linspace(0, 1, num_rules)  # Centers of Gaussian membership functions
sigma = 0.2  # Spread of Gaussian membership functions

# Generate fuzzy rules
def fuzzy_inference(x, centers, sigma, weights):
    rule_outputs = []
    memberships = [gaussian_mf(x, c, sigma) for c in centers]
    total_membership = np.sum(memberships)
    for rule_idx, membership in enumerate(memberships):
        # Apply the rule (TSK consequent is linear)
        rule_output = membership * (np.dot(x, weights[rule_idx, :-1]) + weights[rule_idx, -1])
        rule_outputs.append(rule_output)
    return np.sum(rule_outputs) / total_membership

# Initialize weights for each rule
weights = np.random.uniform(-1, 1, (num_rules, look_back + 1))  # Last weight is bias

# Train the TSK fuzzy system using gradient descent
learning_rate = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        target = y_train[i]

        # Forward pass
        predicted = fuzzy_inference(x, centers, sigma, weights)

        # Error
        error = target - predicted

        # Backpropagation for weights
        memberships = [gaussian_mf(x, c, sigma) for c in centers]
        total_membership = np.sum(memberships)
        for rule_idx, membership in enumerate(memberships):
            membership_scalar = np.mean(membership) / total_membership  # Ensure scalar value
            weights[rule_idx, :-1] += learning_rate * error * membership_scalar * x
            weights[rule_idx, -1] += learning_rate * error * membership_scalar

    # Calculate training error
    train_predictions = [fuzzy_inference(x, centers, sigma, weights) for x in X_train]
    train_error = mean_squared_error(y_train, train_predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training MSE: {train_error}")

# Evaluate on the test set
test_predictions = [fuzzy_inference(x, centers, sigma, weights) for x in X_test]
test_error = mean_squared_error(y_test, test_predictions)
print(f"Test MSE: {test_error}")

# Forecasting future values
forecast_days = 30
forecast_input = data_normalized[-look_back:].reshape(1, -1)
forecast_output = []

for _ in range(forecast_days):
    next_value = fuzzy_inference(forecast_input[0], centers, sigma, weights)
    forecast_output.append(next_value)
    next_value = np.array(next_value).reshape(1, 1)  # Ensure `next_value` is 2D
    forecast_input = np.concatenate((forecast_input[:, 1:], next_value), axis=1)

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
plt.title(f'COVID-19 Forecast for {country_name} using TSK Fuzzy Model')
plt.legend()
plt.grid()
plt.show()
