import json

import dash
import pandas as pd
import requests
from dash import html, dcc, ctx
from dash.dependencies import Input, Output

url = 'https://load-forecast-regressor-cloud-run-service-mqpvakdd5a-ue.a.run.app/forecast'

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    # This is an interval counter ( a scheduler). Configured to increment every 2 seconds. This triggers the call to
    # the model service to get the next forecast and refresh the plot.
    dcc.Interval(id='interval_counter', interval=2 * 1000, n_intervals=0),

    dcc.Graph(id='graph_panel'),

    html.H3(id='actual_label', children=''),  # shows the latest prediction received
    html.H3(id='forecast_label', children=''),  # shows the latest prediction received
    html.Button('Reset Plot', id='reset')
], style={'width': '500'})

# Reading json file
json_file_path = "test_input.json"
with open(json_file_path) as f:
    data = json.load(f)

total_rec_count = len(data)

df_forecast = pd.DataFrame(columns=['x', 'y'])
df_actual = pd.DataFrame(columns=['x', 'y'])


@app.callback([Output('forecast_label', 'children'),
               Output('actual_label', 'children'),
               Output('graph_panel', 'figure')],
              [Input('interval_counter', 'n_intervals'),
               Input('reset', 'n_clicks')],
              prevent_initial_call=True)
def update_graph(n, n_clicks):
    global df_forecast, df_actual

    # if reset button is clicked, reset the graph and the dataframes
    if ctx.triggered_id == 'reset':
        graph_panel = {
            'data': [],
        }
        df_forecast = pd.DataFrame()
        df_actual = pd.DataFrame()
        return '', graph_panel

    idx = n % len(data)

    # send the next record from the test dataset to the model service and get the forecasted load.
    forecast = round(requests.post(url, json=data[idx]).json(), 2)
    actual = round(data[idx]['nat_demand'], 2)
    datetime = data[idx]['datetime']

    # construct a string with the forcast value to display on the dashboard for debugging purposes.
    forecast_label = f'forecast={forecast} MWh'
    actual_label = f'actual={actual} MWh'

    # appending the latest forecast to the global dataframes, so we can pass the series to the plot.
    df_forecast = pd.concat(
        [df_forecast, pd.DataFrame.from_records(data=[(datetime, forecast)], columns=['dt', 'load'])])
    df_actual = pd.concat(
        [df_actual, pd.DataFrame.from_records(data=[(datetime, actual)], columns=['dt', 'load'])])

    # look at the last 14 days' records only, so the scale of the plot stays fixed to a 14-day window.
    last_n = -336
    df_forecast = df_forecast[last_n:]
    df_actual = df_actual[last_n:]

    # preparing the input to the plot. Two lines added, one for the forecast and one for the actual demand (aka
    # nat_demand).
    graph_panel = {
        'data': [
            {
                'x': df_forecast.dt,
                'y': df_forecast.load,
                'name': 'forecast'
            },
            {
                'x': df_actual.dt,
                'y': df_actual.load,
                'name': 'actual'
            }
        ],
        'layout': {
            'margin': {'l': 40, 'r': 0, 't': 40, 'b': 30},
            'title': {'text': 'Panama\'s (Short-term) Electricity Load Forecast - Forecast vs. Actual'},
        },
    }

    return forecast_label, actual_label, graph_panel


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
