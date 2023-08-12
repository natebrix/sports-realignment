import plotly.graph_objs as go
# https://stackoverflow.com/questions/11416024/error-installing-python-snappy-snappy-c-h-no-such-file-or-directory
from PIL import Image
import pandas as pd
import os
import urllib.request

# todo - gen these on first run
# https://github.com/tbryan2/NFL-Python-Team-Logo-Viz/blob/main/Team-Logo-Visualizations.ipynb
def save_logos(logos):
    if not os.path.exists("logos"):
        os.makedirs("logos")
    logo_paths = []
    for team in range(len(logos)):
        urllib.request.urlretrieve(logos['team_logo_espn'][team], f"logos/{logos['team_abbr'][team]}.tif")
        logo_paths.append(f"logos/{logos['team_abbr'][team]}.tif")

def plot_teams(df):

    # Create a scatterplot of the team locations
    data = go.Scattergeo(
        lat = df['team_lat'],
        lon = df['team_lng'],
        mode = 'markers+text',
        marker = dict(
            size = 10,
            color = 'rgba(0, 0, 0, 0.7)',
            line = dict(width = 1, color = 'rgb(255, 255, 255)'),
            symbol = 'circle'
        ),
        text = df['team_abbr'],
        textposition='top center'
    )

    # Set the layout of the plot
    layout = go.Layout(
        geo = dict(
            scope = 'usa',
            projection = dict(type = 'albers usa'),
            showland = True,
            landcolor = 'rgb(250, 250, 250)',
            subunitcolor = 'rgb(217, 217, 217)',
            countrycolor = 'rgb(217, 217, 217)',
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
        title = dict(
            text = 'NFL Team Stadium Locations',
            font = dict(size = 28)
        ),
        margin = dict(r = 0, l = 0, t = 100, b = 0)
    )

    fig = go.Figure(data = [data], layout = layout)

    fig.show()