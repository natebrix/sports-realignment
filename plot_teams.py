import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

def projection_for_league(league):
    if league == 'nfl':
        return dict(type = 'albers usa')
    elif league == 'nba':
        return dict(type = 'albers usa')
    elif league == 'mlb':
        return dict(type = 'albers usa')
    elif league == 'nhl':
        return {"type": "albers", "scale": 2.2}
    else:
        raise ValueError(f"Unknown league {league}")

def plot_divisions(league_name, df, keys=['conf', 'division'], scope='north america', title=None):
    line_color = '#2b0c52'
    data = order_by_division_paths(df, keys)
    data['label'] = data[keys].agg(' '.join, axis=1)
    print(data['label'])
    fig = px.line_geo(data, lon="team_lng", lat="team_lat", scope=scope, line_dash='label', 
                      color='label',
                      text = data['team_abbr'])
    fig.update_traces(line_color=line_color, line_width=5)
    fig.update_traces(textposition='top center')
    proj = projection_for_league(league_name)
    fig.layout = go.Layout(
        geo = dict(
            scope = scope,
            #center = { "lat": 44, "lon": -106.33844897531482 }, # nhl NOOO
            projection = proj,
            showland = True,
            landcolor = 'rgb(250, 250, 250)',
            subunitcolor = 'rgb(177, 177, 177)',
            countrycolor = 'rgb(217, 217, 217)',
            lakecolor = 'rgb(255, 255, 255)',
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
        title = dict(
            text = title,
            font = dict(size = 28)
        ) if title else None
        #margin = dict(r = 0, l = 0, t = 100, b = 0)
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                  width=1500, 
                  height=800, showlegend=True)
    fig.update_layout(legend=dict(
        font = dict(size = 24, color = line_color),
        xanchor="left",
        x=0.0,
        yanchor="top",
        #orientation='h' # hides it for some reason
    ))

    fig.show()
    # fig.write_image("images/fig1.png")
    return fig

def plot_teams(df, scope='north america', title=None):
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
            scope = scope,
            projection = dict(type = 'albers usa'),
            showland = True,
            landcolor = 'rgb(250, 250, 250)',
            subunitcolor = 'rgb(217, 217, 217)',
            countrycolor = 'rgb(217, 217, 217)',
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
        title = dict(
            text = title,
            font = dict(size = 28) 
        ) if title else None,
        margin = dict(r = 0, l = 0, t = 100, b = 0)
    )

    fig = go.Figure(data = [data], layout = layout)
    fig.show()    
    return fig

def plot_max_swaps(df):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(df['swaps'], df['gap'], 'g-', marker='o')  # Added marker for clarity
    ax2.plot(df['swaps'], df['t'], 'b-', marker='o')  # Added marker for clarity

    ax1.set_xlabel('Swaps')
    ax1.set_ylabel('Gap', color='g')
    ax2.set_ylabel('Time', color='b')

    # Set the x-ticks explicitly to match the 'swaps' values
    ax1.set_xticks(df['swaps'][::5])
    ax1.set_xticklabels(df['swaps'][::5])
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.show()