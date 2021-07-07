# LL PRO BONUS: R SHINY APPLICATION ----
# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE | PYTHON DASH ----
# ----

# LIBRARIES

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px

import pandas as pd
import pathlib

# APP SETUP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# PATHS
BASE_PATH = pathlib.Path(__file__).parent.resolve()
ART_PATH = BASE_PATH.joinpath("artifacts").resolve()

# DATA
predictions_df = pd.read_pickle(ART_PATH.joinpath("predictions_df.pkl"))

df = predictions_df \
    .assign(
        spend_actual_vs_pred = lambda x: x['spend_90_total'] - x['pred_spend'] 
    )

# LAYOUT
app.layout = html.Div([
    dcc.Graph(id='graph-slider'),
    dcc.Slider(
        id    = 'spend-slider',
        value = df['spend_actual_vs_pred'].max(),
        max   = df['spend_actual_vs_pred'].max(),
        min   = df['spend_actual_vs_pred'].min()
    )
])

# CALLBACKS 
@app.callback(
    Output('graph-slider', 'figure'),
    Input('spend-slider', 'value'))
def update_figure(spend_delta_max):
    
    df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

    fig = px.scatter(
        data_frame=df_filtered,
        x = 'frequency',
        y = 'pred_prob',
        color = 'spend_actual_vs_pred', 
        color_continuous_midpoint=0, 
        opacity=0.5, 
        color_continuous_scale='IceFire', 
        hover_name='customer_id',
        hover_data=['spend_90_total', 'pred_spend']
    ) \
        .update_layout(
            {
                'plot_bgcolor': 'white'
            }
        )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)