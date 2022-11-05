import dash
import pandas as pd
from dash.dependencies import Input, Output
from dash import html, dcc, ctx
import plotly.graph_objs as go

import json

import requests

url = 'https://load-forecast-regressor-cloud-run-service-mqpvakdd5a-ue.a.run.app/forecast'

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    # This is an interval counter ( a scheduler). Configured to increment every 2 seconds. This triggers the call to
    # the model service to get the next forecast and refresh the plot.
    dcc.Interval(id='interval_counter', interval=2 * 1000, n_intervals=0),

    dcc.Graph(id='graph_panel'),

    html.H3(id='label1', children=''),  # shows the latest prediction received
    html.Button('Reset Plot', id='reset')
], style={'width': '500'})

# Reading json file
json_file_path = "test_input.json"
with open(json_file_path) as f:
    data = json.load(f)

df_forecast = pd.DataFrame(columns=['x', 'y'])
df_actual = pd.DataFrame(columns=['x', 'y'])


# @app.callback([Output('label1', 'children'),
#                Output('graph_panel', 'figure')],
#               [Input('click', 'n_clicks')])
# def reset(n_clicks):
#     if not n_clicks:
#         raise dash.exceptions.PreventUpdate
#     graph_panel = {
#         'data': [],
#         'layout': {
#             'margin': {'l': 40, 'r': 0, 't': 40, 'b': 30},
#             'title': {'text': 'Panama\'s (Short-term) Electricity Load Forecast - Forecast vs. Actual'},
#         },
#     }
#     return graph_panel, ''


@app.callback([Output('label1', 'children'),
               Output('graph_panel', 'figure')],
              [Input('interval_counter', 'n_intervals'), Input('reset', 'n_clicks')],
              prevent_initial_call=True)
def update_graph(n, n_clicks):
    global df_forecast, df_actual

    if ctx.triggered_id == 'reset':
        graph_panel = {
            'data': [],
        }
        df_forecast = pd.DataFrame()
        df_actual = pd.DataFrame()
        return '', graph_panel

    idx = n % len(data)

    # send the next record from the test dataset to the model service and get the forecasted load.
    forecast = requests.post(url, json=data[idx]).json()

    # construct a string with the forcast value to display on the dashboard for debugging purposes.
    forecast_label = f'n={n}, idx={idx} ::   {forecast}'

    # appending the latest forecast to the global dataframes, so we can pass the series to the plot.
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
                'name': 'forecast'
            },
            {
                'x': df_actual.x,
                'y': df_actual.y,
                'name': 'actual'
            }
        ],
        'layout': {
            'margin': {'l': 40, 'r': 0, 't': 40, 'b': 30},
            'title': {'text': 'Panama\'s (Short-term) Electricity Load Forecast - Forecast vs. Actual'},
        },
    }

    return forecast_label, graph_panel


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
