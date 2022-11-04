import dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from dash import html, dcc

import json

from pandas_datareader import data as web
from datetime import datetime as dt

import requests

url = 'https://load-forecast-regressor-cloud-run-service-mqpvakdd5a-ue.a.run.app/forecast'

app = dash.Dash('Load Forecast')

app.layout = html.Div([
    # This is an interval counter ( a scheduler). Configured to increment every 2 seconds. This triggers the call to
    # the model service to get the next forecast and refresh the plot.
    dcc.Interval(id='interval_counter', interval=2 * 1000, n_intervals=0),

    html.H1(id='label1', children=''),  # shows the latest prediction received
    dcc.Graph(id='graph_panel')
], style={'width': '500'})

# Reading json file
json_file_path = "assets/test_input.json"
with open(json_file_path) as f:
    data = json.load(f)

df_forecast = pd.DataFrame(columns=['x', 'y'])
df_actual = pd.DataFrame(columns=['x', 'y'])


@app.callback([Output('label1', 'children'),
               Output('graph_panel', 'figure')],
              [dash.dependencies.Input('interval_counter', 'n_intervals')])
def update_graph(n):
    idx = n % len(data)

    # send the next record from the test dataset to the model service and get the forecasted load.
    forecast = requests.post(url, json=data[idx]).json()

    # construct a string with the forcast value to display on the dashboard for debugging purposes.
    forecast_label = f'n={n}, idx={idx} ::   {forecast}'

    # appending the latest forecast to the global dataframes, so we can pass the series to the plot.
    global df_forecast, df_actual
    df_forecast = pd.concat(
        [df_forecast, pd.DataFrame.from_records(data=[(n, round(forecast, 2))], columns=['x', 'y'])])
    df_actual = pd.concat(
        [df_actual, pd.DataFrame.from_records(data=[(n, round(data[idx]['nat_demand'], 2))], columns=['x', 'y'])])

    # look at the last 14 days' records only, so the scale of the plot stays fixed to a 14-day window.
    last_n = -336
    df_forecast = df_forecast[last_n:]
    df_actual = df_actual[last_n:]

    # preparing the input to the plot. Two lines added, one for the forecast and one for the actual demand (aka
    # nat_demand).
    graph_panel = {
        'data': [
            {
                'x': df_forecast.x,
                'y': df_forecast.y,
            },
            {
                'x': df_actual.x,
                'y': df_actual.y,
            }
        ],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
    }

    return forecast_label, graph_panel


if __name__ == '__main__':
    app.run_server()
