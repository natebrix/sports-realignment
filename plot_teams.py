import plotly.express as px
import plotly.graph_objs as go
# https://stackoverflow.com/questions/11416024/error-installing-python-snappy-snappy-c-h-no-such-file-or-directory
from PIL import Image
import numpy as np
import pandas as pd
import os
import urllib.request

# results rendering
#   show the divisions nicely with helmets
#   show the map with paths

# todo - gen these on first run
# https://github.com/tbryan2/NFL-Python-Team-Logo-Viz/blob/main/Team-Logo-Visualizations.ipynb
def save_logos(logos):
    if not os.path.exists("logos"):
        os.makedirs("logos")
    logo_paths = []
    for team in range(len(logos)):
        urllib.request.urlretrieve(logos['team_logo_espn'][team], f"logos/{logos['team_abbr'][team]}.tif")
        logo_paths.append(f"logos/{logos['team_abbr'][team]}.tif")

# given a dataframe containing lat and long, return an augmented dataset that forms a path between the rows
# that does not cross itself.
#
# we do this by finding the angle of each row from the centroid, and sorting by this angle.
def division_path(d0):
    mc = d0[['team_lng', 'team_lat']].mean()
    ds = list(np.argsort(np.arctan2(d0['team_lng'] - mc['team_lng'], d0['team_lat'] - mc['team_lat'])))
    ds = ds + [ds[0]]
    #print(ds)
    return d0.iloc[ds]

def order_by_division_paths(df, keys):
    ds = [division_path(g[1]) for g in df.groupby(by=keys)]
    return pd.concat(ds)

def plot_divisions(df, keys=['conf', 'division']):
    data = order_by_division_paths(df, keys)
    data['label'] = data[keys].agg(' '.join, axis=1)
    fig = px.line_geo(data, lon="team_lng", lat="team_lat", scope='usa', color='label', 
                      text = data['team_abbr'])
    fig.update_traces(textposition='top center')
    fig.layout = go.Layout(
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
    fig.show()

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