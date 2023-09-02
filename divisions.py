from math import radians, cos, sin, asin, sqrt
import pandas as pd
import gurobipy as gp
#import nfl_data_py as nfl

# x schedule aware based on NFL rules
# competitive balance
# 

# results rendering
#   show the divisions nicely with helmets
#   show the map with paths


#tom = pd.read_csv('data/tom.csv')

def get_locations(teams):
    #logos = nfl.import_team_desc()
    df = pd.read_csv(teams)
    df['team'] = df.index
    #return df.merge(logos, on='team_abbr')
    return df

loc_2002 = get_locations("data/nfl-2002.csv")
loc_2023 = get_locations("data/nfl-2013.csv")

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

def haversine_row(df, i, j):
    lat1, lon1 = df.iloc[i]['team_lat'], df.iloc[i]['team_lng']
    lat2, lon2 = df.iloc[j]['team_lat'], df.iloc[j]['team_lng']
    return haversine(lat1, lon1, lat2, lon2)

def make_distances(df):
    rows = range(df.shape[0])
    return [[haversine_row(df, i, j) for j in rows] for i in rows]

# Define the number of teams and divisions
num_divisions = 8
teams_per_division = 4
confs = {'NFC': ['WEST', 'CENTRAL', 'SOUTH', 'EAST'], 'AFC': ['WEST', 'CENTRAL', 'SOUTH', 'EAST']}
ds = ['NFC_WEST', 'NFC_CENTRAL', 'NFC_SOUTH', 'NFC_EAST', 'AFC_WEST', 'AFC_CENTRAL', 'AFC_SOUTH', 'AFC_EAST'] # todo file


# todo: conf division better
# todo awkward
def score(loc, df_div, team='team_abbr', division='division'):
    df_loc = loc if 'shape' in dir(loc) else pd.read_csv(loc)
    distances = make_distances(df_loc)
    r = {row['team_abbr']:row['team'] for i, row in df_loc.iterrows()}
    return sum([sum([distances[r[i]][r[j]] for i in ts[team] for j in ts[team]]) for (d, ts) in df_div.groupby(division)]) / 2


def base_model_quad(df):
    teams = [row['team_abbr'] for i, row in df.iterrows()] 
    num_teams = df.shape[0]
    distances = make_distances(df)

    m = gp.Model()
    x = {(t, c, d): m.addVar(vtype=gp.GRB.BINARY, name=f"x_{t}_{c}_{d}") for c in confs for d in confs[c] for t in teams}
    y = {(t, c): m.addVar(vtype=gp.GRB.BINARY, name=f"x_{t}_{c}") for c in confs for t in teams}

    sum_division = gp.quicksum(distances[i][j] * x[ai, c, d] * x[aj, c, d] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in confs for d in confs[c])
    sum_same_conf = gp.quicksum(distances[i][j] * y[ai, c] * y[aj, c] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in confs)
    sum_diff_conf = gp.quicksum((distances[i][j] * y[ai, c] * (1 - y[aj, c])) + (distances[i][j] * (1 - y[ai, c]) * y[aj, c])for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in confs)
    #m.setObjective(0.5 * (sum_division + 0.5 * sum_same_conf + 0.25 * sum_diff_conf), gp.GRB.MINIMIZE)
    m.setObjective(0.5 * (sum_division), gp.GRB.MINIMIZE)

    # structural constraints
    for c in confs:
        for d in confs[c]:
            m.addConstr(gp.quicksum(x[a, c, d] for a in teams) == teams_per_division)

    for t in teams:
        m.addConstr(gp.quicksum(x[t, c, d] for c in confs for d in confs[c]) == 1)

    for t in teams:
        for c in confs:
            m.addConstr(gp.quicksum(x[t, c, d] for d in confs[c]) == y[t, c])

    # max swaps
    df_ms = pd.read_csv('opt-data/max_swaps.csv', header=None)
    max_swaps = df_ms.iloc[0, 0]
    fixed_teams = num_teams - max_swaps
    if fixed_teams > 0:
        print(f'MAX SWAPS {max_swaps}')
        # todo brittle
        m.addConstr(gp.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 

    # for everything in fixed.csv, assign x_icd = 1
    fix_teams = pd.read_csv('opt-data/fix_teams.csv')
    for idx, f in fix_teams.iterrows():
        t = f['team_abbr']
        c = f['conf']
        d = f['division']
        print(f'{t} --> {c} {d}')
        m.addConstr(x[t, c, d] == 1)

    # for everything in fix_conf.csv, sum_d x_icd = 1 for fixed c
    fix_conf = pd.read_csv('opt-data/fix_conf.csv')
    for idx, f in fix_conf.iterrows():
        t = f['team_abbr']
        c = f['conf']
        print(f'{t} --> {c}')
        #m.addConstr(gp.quicksum(x[t, c, d] for d in confs[c]) == 1)
        m.addConstr(y[t, c] == 1)

    forbid_teams = pd.read_csv('opt-data/forbid_teams.csv')
    for idx, row in forbid_teams.iterrows():
        i = row['team_abbr1']
        j = row['team_abbr2']
        print(f'{i} != {j}')
        for c in confs:
            for d in confs[c]:
                m.addConstr(x[i, c, d] + x[j, c, d] <= 1)

    m.write('model.mps')
    m.params.NonConvex = 2
    return m, x


def in_division_x(x):
    return x.x > 0.99

def get_assignment(df, x, in_division=in_division_x):
    abbrs = [row['team_abbr'] for i, row in df.iterrows()]
    return [(a, c, d) for c in confs for d in confs[c] for a in abbrs if in_division(x[a, c, d])]

def make_solve_result(a):
    r = pd.DataFrame(a, columns=['team_abbr', 'conf', 'division'])
    return r

def solve(df):
    m, x = base_model_quad(df)
    print(m)
    m.optimize()
    a = get_assignment(df, x)
    return make_solve_result(a)

# There is an assignment problem to assign the division labels.
# This only applies if we are not fixing anything.
# let a_tcd be the inital assignment.
# want to max a_icd x_icd
# I want the permutation of c, d that gives me the best solution
### TODO WRONG
def assign(df, m, x):
    ma = gp.Model()
    z = {(t, c, d): m.addVar(vtype=gp.GRB.BINARY, name=f"z_{t}_{c}_{d}") for (t, c, d) in x}
    m.setObjective(gp.quicksum(x[t, c, d].x * z[t, c, d] for (t, c, d) in x), gp.GRB.MAXIMIZE)
    teams = list(set([t for (t, c, d) in x]))
    for c in confs:
        for d in confs[c]:
            m.addConstr(gp.quicksum(x[t, c, d] for t in teams) == teams_per_division)

