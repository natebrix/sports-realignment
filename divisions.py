from math import radians, cos, sin, asin, sqrt
#import nfl_data_py as nfl
import pandas as pd
import gurobipy as gp

# schedule aware based on NFL rules
# competitive balance
# 

# results rendering
#   show the divisions nicely with helmets
#   show the map with paths

#tom = pd.read_csv('data/tom.csv')

def get_data(teams="data/nfl-2013.csv"):
    #logos = nfl.import_team_desc()
    df = pd.read_csv(teams)
    df['team'] = df.index
    #return df.merge(logos, on='team_abbr')
    return df

df = get_data()

# stackoverflow
def haversine(lat1, lon1, lat2, lon2):
      #R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
      R = 6372.8  # km
      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)

      a = sin(dLat/2)**2 + cos(lat1) * cos(lat2) * sin(dLon/2)**2
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
                distance = haversine(lat1, lon1, lat2, lon2)
            row.append(distance)
        distances.append(row)
    return distances

# Define the number of teams and divisions
num_divisions = 8
teams_per_division = 4
confs = [['NFC_WEST', 'NFC_CENTRAL', 'NFC_SOUTH', 'NFC_EAST'], ['AFC_WEST', 'AFC_CENTRAL', 'AFC_SOUTH', 'AFC_EAST']] # todo file
ds = ['NFC_WEST', 'NFC_CENTRAL', 'NFC_SOUTH', 'NFC_EAST', 'AFC_WEST', 'AFC_CENTRAL', 'AFC_SOUTH', 'AFC_EAST'] # todo file

def score(df, s, team='team_abbr', division='division'):
    distances = make_distances(df)
    r = {row['team_abbr']:row['team'] for i, row in df.iterrows()}
    return sum([sum([distances[r[i]][r[j]] for i in ts[team] for j in ts[team]]) for (d, ts) in s.groupby(division)])

# todo pass in ds
def base_model_quad(df):
    distances = make_distances(df)

    m = gp.Model()

    abbrs = [row['team_abbr'] for i, row in df.iterrows()]
    # todo: think about adding conference
    x = {}
    for a in abbrs:
        for d in ds:
            x[a, d] = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{a}_{d}")

    # z = {}
    # for i in abbrs:
    #     for j in abbrs:
    #         for d in ds:
    #             z[i, j, d] = m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{j}_{d}")

    #y = {}
    #for a in abbrs:
    #    for c in range(len(confs)):
    #        y[a, c] = m.addVar(vtype=gp.GRB.BINARY, name=f"y_{a}_{c}")
    
    # TODO THIS IS NONCONVEX
    # need to linearize
    # z = 𝑥⋅𝑦
    # z ≤ 𝑥⋅𝑈
    # z ≤ 𝑦
    # z ≥ 𝑦−𝑈⋅(1−𝑥)
    # z ≥ 𝑦−1+𝑥

    # for i in abbrs:
    #     for j in abbrs:
    #         for d in ds:
    #             z[i, j, d] = m.addVar(vtype=gp.GRB.BINARY, name=f"z_{i}_{j}_{d}")
    #             m.addConstr(z[i, j, d] <= x[i, d])
    #             m.addConstr(z[i, j, d] <= x[j, d])
    #             m.addConstr(z[i, j, d] >= x[i, d] + x[j, d] - 1)


    # x[ai, d] * x[aj, d] --> z[ai, aj, d]
    m.setObjective(0.5 * gp.quicksum(distances[i][j] * x[ai, d] * x[aj, d] for i, ai in enumerate(abbrs) for j, aj in enumerate(abbrs) for d in ds), gp.GRB.MINIMIZE)
    # m.setObjective(gp.quicksum(distances[i][j] * z[ai, aj, d] for i, ai in enumerate(abbrs) for j, aj in enumerate(abbrs) for d in ds), gp.GRB.MINIMIZE)
    # alternative: distance to each divisional foe + 1/4 distance same conference + 1/8 other conference
    # let y[i, c]
    # d[i, j] * y[i, c] * y[j, c]

    #
    #
    # on the other hand if I have x[i, c, d]
    #  then sum_ijcd (d_ij * x[i, c, d] * x[j, c, d]) + sum_ijc 1/4 * d_ij * sum_d x_icd * sum_d x_jcd
    #
    # or with normal x_id
    #   3/4 sum_ijd d_ij x_id x_jd + 1/4 * sum_ij d_ij (sum_d1 x_id1) (sum_d1 x_jd1) + 1/8
    #for c in range(len(confs)):
    #    m.addConstr(gp.quicksum(y[a, c] for a in abbrs) == len(abbrs) / len(confs))

    #for a in abbrs:
    #    m.addConstr(gp.quicksum(y[a, c] for c in range(len(confs))) == 1)

    # structural constraints
    for d in ds:
        m.addConstr(gp.quicksum(x[a,d] for a in abbrs) == teams_per_division)

    for a in abbrs:
        m.addConstr(gp.quicksum(x[a,d] for d in ds) == 1)

    # max swaps
    max_swaps = pd.read_csv('opt-data/max_swaps.csv', header=None)
    fixed_teams = num_teams - max_swaps.iloc[0,0]
    if fixed_teams > 0:
        m.addConstr(gp.quicksum(x[a, df.loc[i, 'division']] for a in abbrs) >= fixed_teams)

    # for everything in fixed.csv, assign x_id = 1
    fix_teams = pd.read_csv('opt-data/fix_teams.csv')
    for idx, row in fix_teams.iterrows():
        a = row['team_abbr']
        d = row['division']
        m.addConstr(x[a, d] == 1)

    forbid_teams = pd.read_csv('opt-data/forbid_teams.csv')
    for idx, row in forbid_teams.iterrows():
        i = row['team_abbr1']
        j = row['team_abbr2']
        for d in ds:
            m.addConstr(x[i, d] + x[j, d] <= 1)

    m.write('model.mps')
    m.params.NonConvex = 1
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
    print(m)
    m.optimize()
    a = get_assignment(df, x)
    return make_solve_result(a)
