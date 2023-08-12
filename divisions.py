from math import radians, cos, sin, asin, sqrt
import nfl_data_py as nfl
import pandas as pd
import gurobipy as gp

# schedule aware based on NFL rules
# competitive balance
# 

# results rendering
#   show the divisions nicely with helmets
#   show the map with paths

tom = pd.read_csv('tom.csv')

def get_data():
    logos = nfl.import_team_desc()
    df = pd.read_csv("data/teams.csv")
    df['team'] = df.index
    return df.merge(logos, on='team_abbr')

df = get_data()

# stackoverflow
def haversine(lat1, lon1, lat2, lon2):

      R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2 * asin(sqrt(a))

      return R * c


# Compute the distances between teams using the haversine formula
num_teams = df.shape[0]
def make_distances(df):
    distances = []
    for i in range(num_teams):
        row = []
        for j in range(num_teams):
            if i == j:
                distance = 0
            else:
                lat1, lon1 = df.iloc[i]['team_lat'], df.iloc[i]['team_lng']
                lat2, lon2 = df.iloc[j]['team_lat'], df.iloc[j]['team_lng']
                distance = haversine(lon1, lat1, lon2, lat2)
            row.append(distance)
        distances.append(row)
    return distances
distances = make_distances(df)

# Define the number of teams and divisions
num_divisions = 8
teams_per_division = 4
confs = {{'NFC_WEST', 'NFC_CENTRAL', 'NFC_SOUTH', 'NFC_EAST'}, {'AFC_WEST', 'AFC_CENTRAL', 'AFC_SOUTH', 'AFC_EAST'}} # todo file
ds = {'NFC_WEST', 'NFC_CENTRAL', 'NFC_SOUTH', 'NFC_EAST', 'AFC_WEST', 'AFC_CENTRAL', 'AFC_SOUTH', 'AFC_EAST'} # todo file

def score_lists(l):
    return sum([sum([distances[i][j] for i in d for j in d]) for d in l])

def score(r):
    return sum([sum([distances[i][j] for i in ts['team'] for j in ts['team']]) for (d, ts) in r.groupby('division')])

def base_model_quad(df):
    distances = make_distances(df)

    m = gp.Model()

    abbrs = [row['team_abbr'] for i, row in df.iterrows()]
    # todo: think about adding conference
    x = {}
    for a in abbrs:
        for d in ds:
            x[a, d] = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{a}_{d}")
    
    # set objective function
    m.setObjective(gp.quicksum(distances[i][j] * x[ai, d] * x[aj, d] for i, ai in enumerate(abbrs) for j, aj in enumerate(abbrs) for d in ds), gp.GRB.MINIMIZE)

    # structural constraints
    for d in ds:
        m.addConstr(gp.quicksum(x[a,d] for a in abbrs) == teams_per_division)

    for a in abbrs:
        m.addConstr(gp.quicksum(x[a,d] for d in ds) == 1)

    # max swaps
    max_swaps = pd.read_csv('data/max_swaps.csv', header=None)
    fixed_teams = num_teams - max_swaps.iloc[0,0]
    if fixed_teams > 0:
        m.addConstr(gp.quicksum(x[a, df.loc[i, 'division']] for a in abbrs) >= fixed_teams)

    # for everything in fixed.csv, assign x_id = 1
    fix_teams = pd.read_csv('data/fix_teams.csv')
    for idx, row in fix_teams.iterrows():
        a = row['team_abbr']
        d = row['division']
        m.addConstr(x[a, d] == 1)

    forbid_teams = pd.read_csv('data/forbid_teams.csv')
    for idx, row in forbid_teams.iterrows():
        i = row['team_abbr1']
        j = row['team_abbr2']
        for d in ds:
            m.addConstr(x[i, d] + x[j, d] <= 1)

    return m, x

def in_division_x(x, i, d):
    return x[i, d].x > 0.99

def get_assignment(df, x, in_division=in_division_x):
    abbrs = [row['team_abbr'] for i, row in df.iterrows()]
    return [(a, d) for d in ds for a in abbrs if in_division(x, a, d)]

def make_solve_result(a):
    r = pd.DataFrame(a, columns=['team_abbr', 'division'])
    return r

def solve():
    m, x = base_model_quad(df)
    m.optimize()
    a = get_assignment(df, x)
    return make_solve_result(a)
