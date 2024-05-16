import datetime
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
import time
#from solver import SolverRegistery
from gurobimodel import GurobiModel
from scipmodel import ScipModel

def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the haversine distance between two points.
    :return: haversine distance.
    """ 
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


# returns a dictionary of distances between all pairs of teams
def make_distances(df, **args):
    distances = get_arg(args, 'distances', None)
    if distances is not None:
        return distances
    else:
        ll = ['team_abbr', 'team_lat', 'team_lng']
        return {(t[0], t[1]) : t[2] for t in df[ll].merge(df[ll], how='cross').apply(lambda r: haversine_row(r), axis=1).tolist()}

# returns a dictionary of scores for each team
def make_scores(df, **args):
    scores = get_arg(args, 'scores', None)
    if scores is not None:
        return scores
    else:
        return df.set_index('team_abbr')['rating'].to_dict()

# returns a list of columns to use for the objective
def objective_columns(objective):
    if objective[0] == 'c':
        return ['rating']
    elif (objective[0] == 'd') or (objective[0] == 's'):
        return ['team_lat', 'team_lng']
    else:
        raise Exception(f'unknown objective {objective}')

def score_distance_division(div, distances):
    return sum(div.merge(div, how='cross').apply(lambda r: distance_row(r, distances), axis=1).tolist())


def score_distance(df, distances):
    return sum([score_distance_division(div, distances) for (name, div) in df.groupby(['conf', 'division'])]) / 2


def score_competitiveness(df, rating='rating'):
    s = sorted([sum(g[1][rating]) for g in df.groupby(['conf', 'division'])])
    return s[-1] - s[0] # max difference

def get_arg(args, key, default):
    return args[key] if (key in args) else default

# -----

def alignment_matrix(league, r, s):
    sgd = dict(list(r.groupby(['conf', 'division'])))
    ngd = dict(list(s.groupby(['conf', 'division'])))
    divs = league.all_divisions.keys()
    n = len(divs)
    a = np.zeros([n, n])
    for i, (c1, d1) in enumerate(divs):
        for j, (c2, d2) in enumerate(divs):
            a[i, j] = sgd[c1, d1].merge(ngd[c2, d2], on=['team_abbr']).shape[0]
    return a

def get_vars_above_threshold(m, x, threshold):
    return [k for k in x if m.getVal(x[k]) > threshold]

def solve_assignment(d, minimize=True, **kwargs):
    """
    solve an assignment problem with the given row and column ids, with distance matrix d.
    :return: the indexes of the optimal assignment.
    """ 
    n = d.shape[0]
    env = solvers.create_solver_environment(get_arg(kwargs, 'solver', None))
    m = env()
    # note: we should be able to make this continuous, but that depends on the solver returning a basic solution
    # so we will make this binary to be safe.
    # on the other hand this may make the assignment problem intractable.
    #x = {(i, j): m.addContinuousVar(name=f"x_{i}_{j}", lb=0, ub=1) for i in row_ids for j in col_ids}
    x = {(i, j): m.addBinaryVar(name=f"x_{i}_{j}") for i in range(n) for j in range(n)}
    sense = m.minimize if minimize else m.maximize
    m.setObjective(env.quicksum(d[i, j] * x[i, j] for i in range(n) for j in range(n)), sense)
    for i in range(n): 
        m.addConstr(env.quicksum(x[i, j] for j in range(n)) == 1)
    for j in range(n):
        m.addConstr(env.quicksum(x[i, j] for i in range(n)) == 1)
    m.optimize()
    if not m.is_optimal():
        raise Exception('Optimization failed. Not optimal.')
    a = get_vars_above_threshold(m, x, 0.99)
    if len(a) != n: 
        raise Exception(f'Expected {n} assignments, got {len(a)}')
    #print([x[k].x for k in x])
    return a

# realign() pays no regard for incumbent divisions. If no constraints were specified in
# the realignment, we can safely relabel the divisions to best match the incumbent divisions.
def relabel_divisions(league, df_old, df_new, **kwargs):
    # Find the number of shared teams between the realigned and incumbent divisions
    a = alignment_matrix(league, df_new, df_old)

    # find the remapping that preserves the largest number of teams from the incumbent
    sol = solve_assignment(a, minimize=False, **kwargs)

    # now remap
    divs = list(league.all_divisions.keys())
    a_map = [[*divs[i], *divs[j]] for (i, j) in sol]
    dfm = pd.DataFrame(a_map, columns=['conf', 'division', 'new_conf', 'new_division'])
    return dfm.merge(df_new).drop(['conf', 'division'], axis=1).rename({'new_conf':'conf', 'new_division':'division'}, axis=1)

# -----



def get_objective(r, objective, objective_data):
    if objective[0] == 'c':
        return score_competitiveness(r)
    elif (objective[0] == 'd' or objective[0] == 's'):
        return score_distance(r, objective_data)
    else:
        raise Exception(f'unknown objective {objective}')


def realign_result(a, df, keep):
    r = pd.DataFrame(a, columns=['team_abbr', 'conf', 'division'])
    return pd.merge(r, df[['team_abbr'] + keep], on='team_abbr')


def get_algorithms():
    return { "naive" : NaiveModel, "greedy" : GreedyModel, "optimal" : BinlinearModel}


def realign(league, df, objective='distance', algorithm='optimal', **args): 
    if (objective[0] == 'd' or objective[0] == 's'):
        objective_data = make_distances(df, **args)
    elif objective[0] == 'c':
        objective_data = make_scores(df, **args)
    else:
        raise Exception(f'unknown objective {objective}')

    algs = get_algorithms()
    if algorithm not in algs:
        algorithms = ", ".join(algs.keys())
        raise ValueError(f"Unknown algorithm {algorithm}. Choose from {algorithms}.")
    
    solver = algs[algorithm](league, df, **args)
    assign, solve_info = solver.solve(objective, objective_data, **args)
    r = realign_result(assign, df, objective_columns(objective)) 
    solve_info['objective'] = get_objective(r, objective, objective_data)
    solver.log_result(r, solve_info)

    if get_arg(args, 'relabel', False):
        r = relabel_divisions(league, df, r, **args)
    return r, solve_info

