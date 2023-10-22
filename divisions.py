from math import radians, cos, sin, asin, sqrt
import pandas as pd
import gurobipy as gp
#import nfl_data_py as nfl

# MLB here: https://en.wikipedia.org/wiki/Major_League_Baseball

# x schedule aware based on NFL rules
# competitive balance
# 

#tom = pd.read_csv('data/tom.csv')

def get_locations(teams):
    #logos = nfl.import_team_desc()
    df = pd.read_csv(teams)
    df['team'] = df.index
    #return df.merge(logos, on='team_abbr')
    return df

nfl = League.read_csv('data/nfl.csv')
nfl_data = { 2002 : get_locations("data/nfl-2002.csv"),
             2023 : get_locations("data/nfl-2023.csv")
           }

nhl = League.read_csv('data/nhl.csv')
nhl_data = { 
             2023 : get_locations("data/nhl-2023.csv")
           }

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


def haversine_row(r):
    return (r['team_abbr_x'], r['team_abbr_y'], haversine(r['team_lat_x'], r['team_lng_x'], r['team_lat_y'], r['team_lng_y']))


def distance_row(r, distances):  
    return distances[r['team_abbr_x'], r['team_abbr_y']]         


def make_distances(df):
    ll = ['team_abbr', 'team_lat', 'team_lng']
    return {(t[0], t[1]) : t[2] for t in df[ll].merge(df[ll], how='cross').apply(lambda r: haversine_row(r), axis=1).tolist()}


def score_division(div, distances):
    return sum(div.merge(div, how='cross').apply(lambda r: distance_row(r, distances), axis=1).tolist())


def score(teams, distances, team='team_abbr', division='division'):
    return sum([score_division(div, distances) for (name, div) in teams.groupby(['conf', 'division'])]) / 2


def max_swaps_constraints(league, df, m, x, num_teams):
    teams = [row['team_abbr'] for i, row in df.iterrows()] # todo seems bad
    df_ms = pd.read_csv('opt-data/max_swaps.csv', header=None)
    max_swaps = df_ms.iloc[0, 0]
    fixed_teams = num_teams - max_swaps
    if fixed_teams > 0:
        print(f'MAX SWAPS {max_swaps}')
        # todo brittle...why???
        m.addConstr(gp.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 


def fix_division_constraints(league, m, x):
    # for everything in fixed.csv, assign x_icd = 1
    fix_teams = pd.read_csv('opt-data/fix_teams.csv')
    for idx, f in fix_teams.iterrows():
        t = f['team_abbr']
        c = f['conf']
        d = f['division']
        print(f'{t} --> {c} {d}')
        m.addConstr(x[t, c, d] == 1)


def fix_conference_constraints(league, m, y):
     # for everything in fix_conf.csv, sum_d x_icd = 1 for fixed c
    fix_conf = pd.read_csv('opt-data/fix_conf.csv')
    for idx, f in fix_conf.iterrows():
        t = f['team_abbr']
        c = f['conf']
        print(f'{t} --> {c}')
        m.addConstr(y[t, c] == 1)


def forbid_team_constraints(league, m, x):
    forbid_teams = pd.read_csv('opt-data/forbid_teams.csv')
    for idx, row in forbid_teams.iterrows():
        i = row['team_abbr1']
        j = row['team_abbr2']
        print(f'{i} != {j}')
        for c in confs:
            for d in confs[c]:
                m.addConstr(x[i, c, d] + x[j, c, d] <= 1)


def base_model_quad(league, df):
    teams = [row['team_abbr'] for i, row in df.iterrows()] 
    num_teams = df.shape[0]
    distances = make_distances(df)

    m = gp.Model()
    # x_tcd == 1 if team t is in conference c and division d.
    # note, to linearlize can let z_ijcd = x_icd * x_jcd with [appropriate auxiliary constraints]
    x = {(t, c, d): m.addVar(vtype=gp.GRB.BINARY, name=f"x_{t}_{c}_{d}") for (c, d) in league.all_divisions for t in teams}
    # y_tc == 1 if team t is in conference c. That is, some x_tcd == 1.
    y = {(t, c): m.addVar(vtype=gp.GRB.BINARY, name=f"x_{t}_{c}") for c in league.confs for t in teams}

    # todo wrap me in a function to pick the objective.
    sum_division = gp.quicksum(distances[ai, aj] * x[ai, c, d] * x[aj, c, d] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for (c, d) in league.all_divisions)
    sum_same_conf = gp.quicksum(distances[ai, aj] * y[ai, c] * y[aj, c] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in league.confs)
    sum_diff_conf = gp.quicksum((distances[ai, aj] * y[ai, c] * (1 - y[aj, c])) + (distances[ai, aj] * (1 - y[ai, c]) * y[aj, c]) for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in league.confs)
    #m.setObjective(0.5 * (sum_division + 0.5 * sum_same_conf + 0.25 * sum_diff_conf), gp.GRB.MINIMIZE)
    m.setObjective(0.5 * sum_division, gp.GRB.MINIMIZE)

    # structural constraints
    for (c, d) in league.all_divisions:
        m.addConstr(gp.quicksum(x[a, c, d] for a in teams) == league.team_count(c, d))

    for t in teams:
        m.addConstr(gp.quicksum(x[t, c, d] for (c, d) in league.all_divisions) == 1)

    for t in teams:
        for c in league.confs:
            m.addConstr(gp.quicksum(x[t, c, d] for d in league.divisions(c)) == y[t, c])

    max_swaps_constraints(league, df, m, x, num_teams)
    fix_division_constraints(league, m, x)
    fix_conference_constraints(league, m, y)
    forbid_team_constraints(league, m, x)

    m.write('model.mps')
    m.params.NonConvex = 2
    return m, x


def in_division_x(x):
    return x.x > 0.99


def get_assignment(league, df, x, in_division=in_division_x):
    abbrs = [row['team_abbr'] for i, row in df.iterrows()]
    return [(a, c, d) for (c, d) in league.all_divisions for a in abbrs if in_division(x[a, c, d])]


def make_solve_result(a, df, keep=['team_lat', 'team_lng']):
    r = pd.DataFrame(a, columns=['team_abbr', 'conf', 'division'])
    return pd.merge(r, df[['team_abbr'] + keep], on='team_abbr')


def solve(league, df):
    m, x = base_model_quad(league, df)
    print(m)
    m.optimize()
    a = get_assignment(league, df, x)
    return make_solve_result(a, df)
