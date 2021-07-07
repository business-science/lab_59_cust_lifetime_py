# LL PRO BONUS: R SHINY APPLICATION ----
# BUSINESS SCIENCE LEARNING LABS ----
# LAB 59: CUSTOMER LIFETIME VALUE | PYTHON DASH ----
# ----

# LIBRARIES

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

import plotly.express as px

import pandas as pd
import numpy as np

import pathlib

# APP SETUP
external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets
)

PLOT_BACKGROUND = 'rgba(0,0,0,0)'
PLOT_FONT_COLOR = 'white'
LOGO = "https://www.business-science.io/img/business-science-logo.png"

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

# Slider Marks
x = np.linspace(df['spend_actual_vs_pred'].min(), df['spend_actual_vs_pred'].max(), 10, dtype=int)
x = x.round(0)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Customer Spend Prediction", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://www.business-science.io/",
        ),
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        dbc.Collapse(
            id="navbar-collapse", navbar=True, is_open=False
        ),
    ],
    color="dark",
    dark=True,
)

app.layout = html.Div(
    children = [
        navbar, 
        dbc.Row(
            [
                dbc.Col(
                    [
                        
                        html.H3("Welcome to the Customer Analytics Dashboard"),
                        html.Div(
                            id="intro",
                            children="Explore Customers by Predicted Spend versus Actual Spend during the 90-day evaluation period.",
                        ),
                        html.Br(),
                        html.Hr(),
                        html.H5("Spend Actual vs Predicted"),
                        html.P("Segment Customers that were predicted to spend but didn't. Then target these customers with targeted emails."),
                        dcc.Slider(
                            id    = 'spend-slider', 
                            value = df['spend_actual_vs_pred'].max(),
                            max   = df['spend_actual_vs_pred'].max(),
                            min   = df['spend_actual_vs_pred'].min(), 
                            marks = {i: '$'+str(i) for i in range(x[0],x[-1]) if i % 300 == 0}
                        ),
                        html.Br(),
                        html.Button("Download Segmentation", id="btn"), dcc.Download(id="download")
                    ],
                    width = 3,
                    style={'margin':'10px'}
                ),
                dbc.Col(
                    dcc.Graph(id='graph-slider'),
                    width = 8
                )
            ] 
        )
    ]
)

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
        hover_data=['spend_90_total', 'pred_spend'],
    ) \
        .update_layout(
            {
                'plot_bgcolor': PLOT_BACKGROUND,
                'paper_bgcolor':PLOT_BACKGROUND,
                'font_color': PLOT_FONT_COLOR,
                'height':700
            }
        ) \
        .update_traces(
            marker = dict(size = 12)
        )
    
    return fig

# Download Button
@app.callback(
    Output("download", "data"), 
    Input("btn", "n_clicks"), 
    State('spend-slider', 'value'),
    prevent_initial_call=True,
)
def func(n_clicks, spend_delta_max):
    
    df_filtered = df[df['spend_actual_vs_pred'] <= spend_delta_max]

    return dcc.send_data_frame(df_filtered.to_csv, "customer_segmentation.csv")

# Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)