from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
import timeit
import time

# todo: collect and return model info as dictionary

# stackoverflow
# Computes the haversine distance between two points.
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


# returns a dictionary of distances between all pairs of teams
def make_distances(df):
    ll = ['team_abbr', 'team_lat', 'team_lng']
    return {(t[0], t[1]) : t[2] for t in df[ll].merge(df[ll], how='cross').apply(lambda r: haversine_row(r), axis=1).tolist()}

# returns a dictionary of scores for each team
def make_scores(df):
    return df.set_index('team_abbr')['rating'].to_dict()

# returns a list of columns to use for the objective
def objective_columns(objective):
    if objective[0] == 'c':
        return ['rating']
    elif objective[0] == 'd':
        return ['team_lat', 'team_lng']
    else:
        raise Exception(f'unknown objective {objective}')

def score_division(div, distances):
    return sum(div.merge(div, how='cross').apply(lambda r: distance_row(r, distances), axis=1).tolist())


def score(df, distances):
    return sum([score_division(div, distances) for (name, div) in df.groupby(['conf', 'division'])]) / 2


def score_competitiveness(df, rating='rating'):
    s = sorted([sum(g[1][rating]) for g in df.groupby(['conf', 'division'])])
    return s[-1] - s[0] # max difference

def get_arg(args, key, default):
    return args[key] if (key in args) else default


class RealignmentModel:
    def __init__(self, league, df) -> None:
        self.league = league
        self.logfile = get_log_filename('realign')
        self.df = df
        self.algorithm = 'INVALID'

    def log_solve(self, objective, **args):
        with open(self.logfile, 'a') as f:
            f.write(f'**** SOLVE {datetime.now()}\n')
            f.write(f'objective = {objective}\n')
            f.write(f'algorithm = {self.algorithm}\n')
            f.write(f'{self.league.all_divisions}\n')
            for arg in args:
                f.write(f'{arg} = {args[arg]}\n')

    def log(self, msg):
        log_to_file(self.logfile, msg)
        

    def log_result(self, r, solve_info):
        with open(self.logfile, 'a') as f:
            f.write(f'objective = {solve_info["objective"]}\n')
            f.write(f'time = {solve_info["time"]}\n')
            # todo write all scalars in solve_info and shapes of dataframes
            f.write(f'{r.to_csv(index=False)}\n')

    def in_division_x(self, m, x):
        return m.getVal(x) > 0.99


    def get_assignment(self, df, m, x):
        abbrs = [row['team_abbr'] for i, row in df.iterrows()]
        return [(a, c, d) for (c, d) in self.league.all_divisions for a in abbrs if self.in_division_x(m, x[a, c, d])]

    def print_vars(self, m, not_starting_with=[]):
        for v in m.getVars():
            if v.varName[0] not in not_starting_with:
                print(f'{v.varName} = {v.x}')

    # return a dictionary. At minimum it should have the key 'data'
    # with the results of the realignment.
    def solve_core(self, df, objective, objective_data, **args):
        pass

    def solve(self, objective, objective_data, **args):
        self.log_solve(objective, **args)
        start = time.perf_counter()
        result, solve_info = self.solve_core(self.df, objective, objective_data, **args)
        end = time.perf_counter()
        solve_info['time'] = end - start
        return result, solve_info


class NaiveModel(RealignmentModel):
    def __init__(self, league, df, **args) -> None:
        super().__init__(league, df)
        self.algorithm = 'naive'

    # return pre-existing alignment
    def solve_core(self, df, objective, objective_data):
        return df[['team_abbr', 'conf', 'division']].values.tolist(), {}


class GreedyModel(RealignmentModel):
    def __init__(self, league, df, **args) -> None:
        super().__init__(league, df)
        self.algorithm = 'greedy'

    # v: entries
    # k: number of teams in division
    def greedy_dfs_step(self, v, team_count):
        # how to honor constraints...
        # here we can fill in any fixed ones.
        # we can also remove any that are forbidden.
        # max swaps seems hard
        f = v.pop(0)
        t = set(f[0])
        while len(t) < team_count:
            # print(t, len(v))
            i = next(i for i, x in enumerate(v) if (x[0][0] in t) or (x[0][1] in t))
            f = v.pop(i)
            t |= set(f[0]) 
        return list(t)

    def remove_all_in_set(self, v, s):
        return [x for x in v if x[0][0] not in s and x[0][1] not in s] # not efficient


    def get_team_heap(self, objective_data):
        return sorted([x for x in objective_data.items() if x[0][0] < x[0][1]], key=lambda x: x[1])

    # Find a greedy solution in "depth first" order.  That is, fill divisions sequentially.
    def solve_core(self, df, objective, objective_data, **args):
        if objective[0] != 'd':
            raise Exception(f'greedy only works with distance objective')
        # todo: how to do this with competitiveness?
        # --> the greedy choice should be added to the division with the smallest total score
        v = self.get_team_heap(objective_data)
        results = []
        for (c, d) in self.league.all_divisions:
            #print(f'{c} {d}')
            # todo filter v for constraints
            div = self.greedy_dfs_step(v, self.league.team_count(c, d))
            for t in div:
                results.append([t, c, d])
            v = self.remove_all_in_set(v, div)
        return results, {}


def get_model_class(solver):
    if solver == 'gurobi':
        return GurobiModel
    elif solver == 'scip':
        return ScipModel
    else:
        raise Exception(f'unknown solver {solver}')


class BinlinearModel(RealignmentModel):
    def __init__(self, league, df, **args) -> None:
        super().__init__(league, df)
        solver = get_arg(args, 'solver', 'gurobi')
        self.model = get_model_class(solver)
        self.algorithm = 'bilinear'
        self.args = args

    def max_swaps_constraints(self, solve_info, df, m, x, num_teams, max_swaps):
        teams = df['team_abbr'].unique() 
        if max_swaps is None:
            df_ms = pd.read_csv('opt-data/max_swaps.csv', header=None)
            max_swaps = df_ms.iloc[0, 0]
        solve_info['max_swaps'] = max_swaps
        fixed_teams = num_teams - max_swaps
        if fixed_teams > 0:
            # todo brittle...why???
            # I want the sum of the x_icd for the existing assignment to be at least BLAH
            # todo this is brittle because how do I now that order of teams matches order of df?
            m.addConstr(self.model.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 


    def fix_division_constraints(self, solve_info, m, x):
        # for everything in fixed.csv, assign x_icd = 1
        fix_teams = pd.read_csv('opt-data/fix_teams.csv')
        if fix_teams.shape[0] > 0:
            solve_info['fix_teams'] = fix_teams
        for idx, f in fix_teams.iterrows():
            t = f['team_abbr']
            c = f['conf']
            d = f['division']
            print(f'{t} --> {c} {d}')
            m.addConstr(x[t, c, d] == 1)

    def fix_conference_constraints(self, solve_info, m, y):
        # for everything in fix_conf.csv, sum_d x_icd = 1 for fixed c
        fix_conf = pd.read_csv('opt-data/fix_conf.csv')
        if fix_conf.shape[0] > 0:
            solve_info['fix_conf'] = fix_conf

        for idx, f in fix_conf.iterrows():
            t = f['team_abbr']
            c = f['conf']
            print(f'{t} --> {c}')
            m.addConstr(y[t, c] == 1)


    def forbid_team_constraints(self, solve_info, m, x):
        forbid_teams = pd.read_csv('opt-data/forbid_teams.csv')
        if forbid_teams.shape[0] > 0:
            solve_info['forbid_teams'] = forbid_teams
        for idx, row in forbid_teams.iterrows():
            i = row['team_abbr1']
            j = row['team_abbr2']
            print(f'{i} != {j}')
            for c in self.league.confs:
                for d in self.league.confs[c]:
                    m.addConstr(x[i, c, d] + x[j, c, d] <= 1)


    def structural_constraints(self, solve_info, teams, m, x, y):
        for (c, d) in self.league.all_divisions:
            m.addConstr(self.model.quicksum(x[a, c, d] for a in teams) == self.league.team_count(c, d))

        for t in teams:
            m.addConstr(self.model.quicksum(x[t, c, d] for (c, d) in self.league.all_divisions) == 1)

        for t in teams:
            for c in self.league.confs:
                m.addConstr(self.model.quicksum(x[t, c, d] for d in self.league.divisions(c)) == y[t, c])


    def competitiveness_objective(self, solve_info, teams, s, m, x):
        # if we are trying to maximize competitiveness then what we want is to minimize the maximum 
        # gap in strengths between divisions.
        r = {(c, d): m.addContinuousVar(name=f"r_{c}_{d}", lb=float('-inf')) for (c, d) in self.league.all_divisions}
        for (c, d) in self.league.all_divisions:
            m.addConstr(r[c, d] == self.model.quicksum(x[t, c, d] * s[t] for t in teams))
        rm = m.addContinuousVar(name="rm", lb=float('-inf'))
        m.addConstrs(rm >= r[c, d] - r[c2, d2] for (c, d) in self.league.all_divisions for (c2, d2) in self.league.all_divisions)
        m.setObjective(rm, self.model.minimize)

    # todo stability objective
    def stability_objective(self, solve_info, teams, s, m, x):
        # we want to minimize the number of teams that change divisions
        # we want an indicator variable for each team that is 1 if the team changes divisions
        u = {t: m.addBinaryVar(name=f"u_{t}") for t in teams}
        # df.loc[i, 'conf'], df.loc[i, 'division']]
        # todo df?!?!
        m.addConstr(m.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 


    def distance_objective(self, solve_info, teams, distances, m, x, y, **args):
        linearize = get_arg(args, 'linearize', False)
        dummy = get_arg(args, 'dummy', False)
        solve_info['linearize'] = linearize
        if linearize:
            # z_t1t2cd == 1 if teams t1 and t2 are in conference c and division d.
            z = {(t1, t2, c, d): m.addBinaryVar(name=f"z_{t1}_{t2}_{c}_{d}") for (c, d) in self.league.all_divisions for t1 in teams for t2 in teams}
            for (c, d) in self.league.all_divisions:
                for t1 in teams:
                    for t2 in teams:
                        m.addConstr(z[t1, t2, c, d] <= x[t1, c, d])
                        m.addConstr(z[t1, t2, c, d] >= x[t1, c, d] + x[t2, c, d] - 1)

        if linearize:
            sum_division = self.model.quicksum(distances[ai, aj] * z[ai, aj, c, d] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for (c, d) in self.league.all_divisions)
        else:
            sum_division = self.model.quicksum(distances[ai, aj] * x[ai, c, d] * x[aj, c, d] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for (c, d) in self.league.all_divisions)
        sum_same_conf = self.model.quicksum(distances[ai, aj] * y[ai, c] * y[aj, c] for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in self.league.confs)
        sum_diff_conf = self.model.quicksum((distances[ai, aj] * y[ai, c] * (1 - y[aj, c])) + (distances[ai, aj] * (1 - y[ai, c]) * y[aj, c]) for i, ai in enumerate(teams) for j, aj in enumerate(teams) for c in self.league.confs)
        
        obj = 0.5 * sum_division
        if dummy:
            dummy_obj = m.addContinuousVar("cost")
            m.setObjective(dummy_obj, self.model.minimize)
            m.addConstr(dummy_obj == obj)
        else:
            m.setObjective(obj, self.model.minimize)

    def bilinear_start(self, solve_info, df, m, x, y):
        solution = m.createSol()
        m.update() 

        for x_tcd in x.values():
            m.setSolVal(solution, x_tcd, 0) 

        for y_tc in y.values():
            m.setSolVal(solution, y_tc, 0) 

        for (c, d), div in df.groupby(['conf', 'division']):
            for i, row in div.iterrows():
                t = row['team_abbr']
                m.setSolVal(solution, x[t, c, d], 1) 
                m.setSolVal(solution, y[t, c], 1) 
        
        # Certain solvers cannot handle nonconvex objectives directly. In such a case, the solver wrapper
        # internally adds a dummy variable to the model and sets the objective to minimize this variable.
        # Therefore this dummy variables must be warmstarted also.
        if m.isNonconvex():
            m.setNonconvexSolVal(solution)
        m.addSol(solution)

    def create_model_bilinear(self, df, objective, objective_data, **args):
        self.log(f'solver = {self.model.__name__}\n')
        solve_info = {}

        linearize = get_arg(args, 'linearize', False)
        max_swaps = get_arg(args, 'max_swaps', None)
        warm = get_arg(args, 'warm', False)

        # todo: do some kind of check to make sure df is compatible with league
        teams = [row['team_abbr'] for i, row in df.iterrows()] 
        num_teams = df.shape[0]

        m = self.model()

        # todo use solver_params. todo make solver agnostic - not sure how to do this since this is inherently solver specific
        # m.setParam('Method', 2)       # 0 - primal simplex, 1 - dual simplex, 2 - barrier, etc.
        # m.setParam('Heuristics', 0.0) # Off - 0.0, Default - 0.05
        # m.setParam('Cuts', 1)         # -1 - automatic, 0 - off, 1 - conservative, 2 - aggressive
        # m.setParam('CutPasses', 5)    # Maximum number of cutting plane passes (-1 for no limit)
        # m.setParam('Presolve', 2)     # -1 - automatic, 0 - off, 1 - conservative, 2 - aggressive
        #m.setParam("lp/resolvealgorithm", "b") 
        #m.setParam("separating/maxrounds", 1)
        #m.setParam("separating/maxroundsroot", 5)
        #m.setParam("presolving/maxrestarts", 2)

        m.setNonconvex(not linearize)
        m.setLogFile(self.logfile) 

        # I could introduce a variable w_tu which is 1 iff t and u are in the same division.
        # then x_tcd + x_ucd - 1 <= w_tu for all t, u, c, d

        # x_tcd == 1 if team t is in conference c and division d.
        x = {(t, c, d): m.addBinaryVar(name=f"x_{t}_{c}_{d}") for (c, d) in self.league.all_divisions for t in teams}

        # y_tc == 1 if team t is in conference c. That is, some x_tcd == 1.
        y = {(t, c): m.addBinaryVar(name=f"x_{t}_{c}") for c in self.league.confs for t in teams}

        if objective[0] == 'd':
            self.distance_objective(solve_info, teams, objective_data, m, x, y, **self.args)
        elif objective[0] == 'c':
            self.competitiveness_objective(solve_info, teams, objective_data, m, x)
        else:
            raise Exception(f'unknown objective {objective}')

        self.structural_constraints(solve_info, teams, m, x, y)
        self.max_swaps_constraints(solve_info, df, m, x, num_teams, max_swaps)
        self.fix_division_constraints(solve_info, m, x)
        self.fix_conference_constraints(solve_info, m, y)
        self.forbid_team_constraints(solve_info, m, x)

        solve_info['warm'] = warm
        if warm and not linearize:
            self.bilinear_start(solve_info, df, m, x, y)

        if args.get('mps_file'):
            m.write(args['mps_file'])

        return m, x, solve_info

    def solve_core(self, df, objective, objective_data, **args):
        m, x, solve_info = self.create_model_bilinear(df, objective, objective_data, **args)
        m.optimize()
        solve_info['solve_time'] = m.getSolvingTime()
        if not m.is_optimal():
            raise Exception(f'solver did not find optimal solution')    
        assign = self.get_assignment(df, m, x)
        if get_arg(args, 'verbose', False):
            self.print_vars(m, ['x', 'y'])
        return assign, solve_info


def get_objective(r, objective, objective_data):
    if objective[0] == 'c':
        return score_competitiveness(r)
    elif objective[0] == 'd':
        return score(r, objective_data)
    else:
        raise Exception(f'unknown objective {objective}')

def realign_result(a, df, keep):
    r = pd.DataFrame(a, columns=['team_abbr', 'conf', 'division'])
    return pd.merge(r, df[['team_abbr'] + keep], on='team_abbr')


def get_algorithms():
    return { "naive" : NaiveModel, "greedy" : GreedyModel, "optimal" : BinlinearModel}

def realign(league, df, objective='distance', algorithm='optimal', **args): 
    if objective[0] == 'd':
        objective_data = make_distances(df)
    elif objective[0] == 'c':
        objective_data = make_scores(df)
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
    return r, solve_info

# todo simple heuristic to assign reasonable names to divisions for NFL.
# Let Seattle be in the NFC WEST.
# Then take the farthest west centroid division and call that AFC WEST.
# Then let New York Giants be in the NFC EAST.
# Then take the farthest east centroid division and call that AFC EAST.
# Then take the two farthest north divisions. Call one NFC NORTH and the other AFC NORTH.
# and do the same for south. 
