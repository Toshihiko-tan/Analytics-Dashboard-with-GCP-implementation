# This is the main module that imports the two modules and runs the functions

import data_cleaning_module
import visualization_and_prediction_module
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import numpy as np
from data_cleaning_module import df


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Hello Dash'),

    html.Div('''
        Dash: A web application framework for your data. 
    '''),

    dcc.Graph(
        id='example-graph',
        # figure=fig
    ),

    html.Div('''
        Dash: Another example for chart
    '''),

    dcc.Graph(
        id='example-graph2',
        # figure=fig2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
