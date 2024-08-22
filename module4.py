
from cgi import test
from pickle import NONE
from sre_parse import CATEGORIES
from pandas.tseries.offsets import Second
import prophet
from prophet.plot import plot_components_plotly, plot_plotly

import wget
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import Input, Output, dcc, html
import plotly.graph_objs as go
import numpy as np
import datetime

df = pd.read_csv('feature.csv')

# Data preparation
df['Date'] = pd.to_datetime(df['Date'])
prophet_df = df[['Date', 'Temperature']].copy()
prophet_df.rename(columns={'Date': 'ds', 'Temperature': 'y'}, inplace=True)
prophet_df.dropna(inplace=True)
prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
prophet_df.dropna(subset=['y'], inplace=True)

# Visualize data
plt.figure(figsize=(12, 6))
plt.plot(prophet_df['ds'], prophet_df['y'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Over Time')
plt.show()

# Initialize and fit model
model = Prophet()
model.fit(prophet_df)

# Create future DataFrame
forecast_periods = 60
future = model.make_future_dataframe(periods=forecast_periods)

# Make predictions
forecast = model.predict(future)

# Visualize forecast
model.plot(forecast)
plt.title('Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()

# Visualize forecast components
model.plot_components(forecast)
plt.show()



   







