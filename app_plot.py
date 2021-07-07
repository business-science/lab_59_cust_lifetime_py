import plotly.express as px

import pandas as pd

predictions_df = pd.read_pickle('artifacts/predictions_df.pkl')

df = predictions_df \
    .assign(
        spend_actual_vs_pred = lambda x: x['spend_90_total'] - x['pred_spend'] 
    )

px.scatter(
    data_frame=df,
    x = 'frequency',
    y = 'pred_prob',
    color = 'spend_actual_vs_pred', 
    color_continuous_midpoint=0, 
    opacity=0.5, 
    color_continuous_scale='IceFire', 
    # trendline='lowess', 
    # trendline_color_override='black'
) \
    .update_layout(
        {
            'plot_bgcolor': 'white'
        }
    )

