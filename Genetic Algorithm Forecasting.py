%reset -f

# Genetic Algorithm Forecasting

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
data = country_data.values.reshape(-1, 1)  # Reshape to a 2D array for GA

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data for Genetic Algorithm
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

# Genetic Algorithm parameters
population_size = 100
generations = 100
mutation_rate = 0.2

# Number of features (weights for linear regression)
num_features = look_back

# Objective function: Calculate fitness (negative MSE)
def calculate_fitness(population):
    fitness = []
    for individual in population:
        predictions = np.dot(X_train, individual)
        mse = mean_squared_error(y_train, predictions)
        fitness.append(-mse)  # Negative because we minimize MSE
    return np.array(fitness)

# Crossover: Combine genes from two parents
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_features)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# Mutation: Randomly change some genes
def mutate(individual):
    for i in range(num_features):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.uniform(-0.5, 0.5)  # Add small random value
    return individual

# Initialize population
population = [np.random.uniform(-1, 1, num_features) for _ in range(population_size)]

# Genetic Algorithm process
for generation in range(generations):
    fitness = calculate_fitness(population)
    best_individual = population[np.argmax(fitness)]
    print(f"Generation {generation + 1}: Best Fitness = {-np.max(fitness)}")  # Print best MSE

    # Select parents based on fitness
    probabilities = fitness - np.min(fitness) + 1e-6  # Normalize probabilities to avoid negatives
    probabilities /= np.sum(probabilities)
    # Convert population to an array of objects to work with np.random.choice
    population_array = np.array(population, dtype=object)
    parents = [
        (
            population_array[np.random.choice(range(len(population_array)), p=probabilities)],
            population_array[np.random.choice(range(len(population_array)), p=probabilities)],
            )
        for _ in range(population_size)
        ]


    # Generate new population
    new_population = []
    for parent1, parent2 in parents:
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    population = new_population

# Best solution
best_weights = best_individual

# Forecasting future values
# Forecasting future values
forecast_days = 50
forecast_input = data_normalized[-look_back:].reshape(1, -1)
forecast_output = []

for _ in range(forecast_days):
    next_value = np.dot(forecast_input, best_weights)  # Use optimized weights
    forecast_output.append(next_value)
    next_value = np.array(next_value).reshape(1, 1)  # Reshape `next_value` to 2D
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
plt.title(f'COVID-19 Forecast for {country_name} using Genetic Algorithm')
plt.legend()
plt.grid()
plt.show()
