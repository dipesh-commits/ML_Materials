import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
import matplotlib.pyplot as plt

columns= ['code','country', 'ds', 'action', 'y']

df = pd.read_csv('my_test/test/export.csv', names= columns)


my_columns = ['ds','y']

new_df = pd.DataFrame(df, columns= my_columns)
# print(new_df['ds'].unique())
# print(new_df.groupby('ds')['y'].sum())

df = df.groupby(["ds", "y"], as_index=False, sort=False).count()
idx = df.groupby("ds", sort=False).transform(max)["country"] == df["country"]
df = df[idx][["ds", "y"]].reset_index(drop=True)
print(df.head())
m = Prophet()
m.fit(df)
print(m)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)


# py.init_notebook_mode()
#
# fig = plot_plotly(m, forecast)  # This returns a plotly Figure
# py.plot(fig)

