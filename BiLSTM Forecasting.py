%reset -f

# Bi-LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'time_series_covid19_confirmed_global.csv'
covid_data = pd.read_csv(file_path)

# Select the country (Iran in this case)
country_name = 'Iran'
country_data = covid_data[covid_data['Country/Region'] == country_name]

# Aggregate data for the country (if multiple provinces exist)
country_data = country_data.iloc[:, 4:].sum()  # Skip Lat, Long, and metadata
country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y')
data = country_data.values.reshape(-1, 1)  # Reshape to a 2D array for BiLSTM

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Prepare data for BiLSTM
look_back = 50  # Number of previous days to use for prediction

X, y = [], []
for i in range(len(data_normalized) - look_back):
    X.append(data_normalized[i:i + look_back, 0])
    y.append(data_normalized[i + look_back, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for BiLSTM input

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build BiLSTM model
model = Sequential([
    Bidirectional(LSTM(50, activation='relu', input_shape=(look_back, 1))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Forecast future values
forecast_days = 100
forecast_input = data_normalized[-look_back:].reshape(1, look_back, 1)
forecast_output = []

for _ in range(forecast_days):
    next_value = model.predict(forecast_input, verbose=0)
    forecast_output.append(next_value[0, 0])  # Append the scalar value
    # Update the forecast input by removing the oldest value and appending the new prediction
    forecast_input = np.append(forecast_input[:, 1:, :], [[next_value[0]]], axis=1)

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
plt.title(f'COVID-19 Forecast for {country_name} using BiLSTM')
plt.legend()
plt.grid()
plt.show()
