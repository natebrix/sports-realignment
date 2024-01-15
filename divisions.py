from math import radians, cos, sin, asin, sqrt
import pandas as pd
from datetime import datetime
import timeit
#tom = pd.read_csv('data/tom.csv')

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

def make_scores(df):
    return df.set_index('team_abbr')['rating'].to_dict()


def objective_columns(objective):
    if objective[0] == 'c':
        return ['rating']
    elif objective[0] == 'd':
        return ['team_lat', 'team_lng']
    else:
        raise Exception(f'unknown objective {objective}')


def score_division(div, distances):
    return sum(div.merge(div, how='cross').apply(lambda r: distance_row(r, distances), axis=1).tolist())


def score(teams, distances, team='team_abbr', division='division'):
    return sum([score_division(div, distances) for (name, div) in teams.groupby(['conf', 'division'])]) / 2

class Realign:
    def __init__(self, league, model) -> None:
        self.league = league
        self.model = model
        suffix = datetime.now().strftime("%Y_%m_%d")
        self.logfile = f'realign_{suffix}.log'

    def log(self, msg):
        with open(self.logfile, 'a') as f:
            f.write(f'{msg}\n')

    def log_solve(self, objective, **args):
        with open(self.logfile, 'a') as f:
            f.write(f'**** SOLVE {datetime.now()}\n')
            f.write(f'objective = {objective}\n')
            f.write(f'{self.league.all_divisions}\n')
            for arg in args:
                f.write(f'{arg} = {args[arg]}\n')

    def max_swaps_constraints(self, df, m, x, num_teams):
        teams = df['team_abbr'].unique() # [row['team_abbr'] for i, row in df.iterrows()] # todo seems bad
        df_ms = pd.read_csv('opt-data/max_swaps.csv', header=None)
        max_swaps = df_ms.iloc[0, 0]
        fixed_teams = num_teams - max_swaps
        if fixed_teams > 0:
            print(f'MAX SWAPS {max_swaps}')
            # todo brittle...why???
            m.addConstr(m.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 


    def fix_division_constraints(self, m, x):
        # for everything in fixed.csv, assign x_icd = 1
        fix_teams = pd.read_csv('opt-data/fix_teams.csv')
        for idx, f in fix_teams.iterrows():
            t = f['team_abbr']
            c = f['conf']
            d = f['division']
            print(f'{t} --> {c} {d}')
            m.addConstr(x[t, c, d] == 1)


    def fix_conference_constraints(self, m, y):
        # for everything in fix_conf.csv, sum_d x_icd = 1 for fixed c
        fix_conf = pd.read_csv('opt-data/fix_conf.csv')
        for idx, f in fix_conf.iterrows():
            t = f['team_abbr']
            c = f['conf']
            print(f'{t} --> {c}')
            m.addConstr(y[t, c] == 1)


    def forbid_team_constraints(self, m, x):
        forbid_teams = pd.read_csv('opt-data/forbid_teams.csv')
        for idx, row in forbid_teams.iterrows():
            i = row['team_abbr1']
            j = row['team_abbr2']
            print(f'{i} != {j}')
            for c in self.league.confs:
                for d in self.league.confs[c]:
                    m.addConstr(x[i, c, d] + x[j, c, d] <= 1)


    def structural_constraints(self, teams, m, x, y):
        for (c, d) in self.league.all_divisions:
            m.addConstr(self.model.quicksum(x[a, c, d] for a in teams) == self.league.team_count(c, d))

        for t in teams:
            m.addConstr(self.model.quicksum(x[t, c, d] for (c, d) in self.league.all_divisions) == 1)

        for t in teams:
            for c in self.league.confs:
                m.addConstr(self.model.quicksum(x[t, c, d] for d in self.league.divisions(c)) == y[t, c])


    def competitiveness_objective(self, teams, s, m, x):
        # if we are trying to maximize competitiveness then what we want is to minimize the maximum 
        # gap in strengths between divisions.
        r = {(c, d): m.addContinuousVar(name=f"r_{c}_{d}", lb=float('-inf')) for (c, d) in self.league.all_divisions}
        for (c, d) in self.league.all_divisions:
            m.addConstr(r[c, d] == self.model.quicksum(x[t, c, d] * s[t] for t in teams))
        rm = m.addContinuousVar(name="rm", lb=float('-inf'))
        m.addConstrs(rm >= r[c, d] - r[c2, d2] for (c, d) in self.league.all_divisions for (c2, d2) in self.league.all_divisions)
        m.setObjective(rm, self.model.minimize)

    # todo stability objective
    def stability_objective(self, teams, s, m, x):
        # we want to minimize the number of teams that change divisions
        # we want an indicator variable for each team that is 1 if the team changes divisions
        u = {t: m.addBinaryVar(name=f"u_{t}") for t in teams}
        # df.loc[i, 'conf'], df.loc[i, 'division']]
        m.addConstr(m.quicksum(x[t, df.loc[i, 'conf'], df.loc[i, 'division']] for i, t in enumerate(teams)) >= fixed_teams) 


    def distance_objective(self, teams, distances, m, x, y, **args):
        linearize = self.get_arg(args, 'linearize', False)
        dummy = self.get_arg(args, 'dummy', False)

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

    def get_arg(self, args, key, default):
        return args[key] if args.get(key) else default

    def base_model_quad(self, df, objective='distance', **args):
        linearize = self.get_arg(args, 'linearize', False)
        # todo: do some kind of check to make sure df is compatible with league
        teams = [row['team_abbr'] for i, row in df.iterrows()] 
        num_teams = df.shape[0]

        m = self.model()
        m.setNonconvex(not linearize)
        m.setLogFile(self.logfile) # todo write all the other data to this too

        # I could introduce a variable w_tu which is 1 iff t and u are in the same division.
        # then x_tcd + x_ucd - 1 <= w_tu for all t, u, c, d

        # x_tcd == 1 if team t is in conference c and division d.
        x = {(t, c, d): m.addBinaryVar(name=f"x_{t}_{c}_{d}") for (c, d) in self.league.all_divisions for t in teams}

        # y_tc == 1 if team t is in conference c. That is, some x_tcd == 1.
        y = {(t, c): m.addBinaryVar(name=f"x_{t}_{c}") for c in self.league.confs for t in teams}

        if objective[0] == 'd':
            distances = make_distances(df)
            self.distance_objective(teams, distances, m, x, y, **args)
        elif objective[0] == 'c':
            scores = make_scores(df)
            self.competitiveness_objective(teams, scores, m, x)
        else:
            raise Exception(f'unknown objective {objective}')

        self.structural_constraints(teams, m, x, y)
        self.max_swaps_constraints(df, m, x, num_teams)
        self.fix_division_constraints(m, x)
        self.fix_conference_constraints(m, y)
        self.forbid_team_constraints(m, x)

        if args.get('mps_file'):
            m.write(args['mps_file'])

        return m, x


    def in_division_x(self, x):
        return x.x > 0.99


    def get_assignment(self, df, x):
        abbrs = [row['team_abbr'] for i, row in df.iterrows()]
        return [(a, c, d) for (c, d) in self.league.all_divisions for a in abbrs if self.in_division_x(x[a, c, d])]


    def make_solve_result(self, a, df, keep):
        r = pd.DataFrame(a, columns=['team_abbr', 'conf', 'division'])
        return pd.merge(r, df[['team_abbr'] + keep], on='team_abbr')

    def print_vars(self, m, not_starting_with=[]):
        for v in m.getVars():
            if v.varName[0] not in not_starting_with:
                print(f'{v.varName} = {v.x}')

    def solve(self, df, objective='distance', **args):
        self.log_solve(objective, **args)
        m, x = self.base_model_quad(df, objective, **args)
        t = timeit.timeit(m.optimize, number=1)
        self.log(f'elapsed time = {t}')
        a = self.get_assignment( df, x)
        if self.get_arg(args, 'verbose', False):
            self.print_vars(m, ['x', 'y'])
        r = self.make_solve_result(a, df, objective_columns(objective))
        self.log(str(r))
        return r

