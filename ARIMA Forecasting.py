%reset -f

# ARIMA (AutoRegressive Integrated Moving Average)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Load the dataset
file_path = 'time_series_covid19_confirmed_global.csv'
covid_data = pd.read_csv(file_path)

# Select the country (Iran in this case)
country_name = 'Iran'
country_data = covid_data[covid_data['Country/Region'] == country_name]

# Aggregate data for the country (if multiple provinces exist)
country_data = country_data.iloc[:, 4:].sum()  # Skip Lat, Long, and metadata
country_data.index = pd.to_datetime(country_data.index, format='%m/%d/%y')

# Forecasting parameters
forecast_days = 365  # Desired number of future steps

# Train ARIMA model
model = ARIMA(country_data, order=(5, 1, 0))  # ARIMA(p,d,q)
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_days)

# Create a timeline for the forecast
forecast_index = pd.date_range(start=country_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

# Plot historical data
plt.figure(figsize=(12, 6))
plt.plot(country_data, label='Historical Data', marker='o')
plt.plot(forecast_index, forecast, label='Forecasted Data', linestyle='--', marker='.', color='red')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.title(f'COVID-19 Forecast for {country_name}')
plt.legend()
plt.grid()
plt.show()
