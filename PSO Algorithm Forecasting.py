%reset -f

# PSO Algorithm Forecasting

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
data = country_data.values.reshape(-1, 1)  # Reshape to a 2D array for PSO

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data for PSO
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

# PSO parameters
num_particles = 100
num_iterations = 100
inertia = 0.5
cognitive_param = 1.5
social_param = 1.5

# Initialize particle positions and velocities
num_features = look_back
particle_positions = np.random.uniform(-1, 1, (num_particles, num_features))
particle_velocities = np.random.uniform(-0.1, 0.1, (num_particles, num_features))
personal_best_positions = particle_positions.copy()
personal_best_scores = np.array([float('inf')] * num_particles)
global_best_position = None
global_best_score = float('inf')

# Objective function: Mean Squared Error
def evaluate_fitness(position):
    predictions = np.dot(X_train, position)
    mse = mean_squared_error(y_train, predictions)
    return mse

# PSO Optimization
for iteration in range(num_iterations):
    for i in range(num_particles):
        fitness = evaluate_fitness(particle_positions[i])
        # Update personal best
        if fitness < personal_best_scores[i]:
            personal_best_scores[i] = fitness
            personal_best_positions[i] = particle_positions[i]
        # Update global best
        if fitness < global_best_score:
            global_best_score = fitness
            global_best_position = particle_positions[i]

    # Update velocities and positions
    for i in range(num_particles):
        r1, r2 = np.random.random(num_features), np.random.random(num_features)
        cognitive_component = cognitive_param * r1 * (personal_best_positions[i] - particle_positions[i])
        social_component = social_param * r2 * (global_best_position - particle_positions[i])
        particle_velocities[i] = inertia * particle_velocities[i] + cognitive_component + social_component
        particle_positions[i] += particle_velocities[i]

    print(f"Iteration {iteration + 1}: Best Fitness = {global_best_score}")

# Use global best position as the optimal weights
optimal_weights = global_best_position

# Forecasting future values
forecast_days = 50
forecast_input = data_normalized[-look_back:].reshape(1, -1)
forecast_output = []

for _ in range(forecast_days):
    next_value = np.dot(forecast_input, optimal_weights)  # Use optimized weights
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
plt.title(f'COVID-19 Forecast for {country_name} using PSO')
plt.legend()
plt.grid()
plt.show()
