# Routines for reproducing paper results.

from datetime import datetime
import timeit
import numpy as np
import pandas as pd
import subprocess

solvers = SolverRegistry([ScipModel, GurobiModel])
scip = solvers.create_solver_environment('scip')

all_algs = list(get_algorithms().keys())

def get_path_incumbent(league_name, season):
    return f'data/{league_name}-{season}.csv'

def get_path_optimal(league_name, season):
    return f'out/{league_name}_{season}_distance_optimal.csv'

def write_by_conference(league, season):
    df = league.seasons[season]
    g = df.groupby('conf')
    for conf in g.groups:
        df_c = g.get_group(conf)
        filename = f'{league.name.lower()}-{season}-{conf.lower()}.csv' 
        print(f"Writing {filename}")
        df_c.to_csv(f'data/{filename}', index=False)

def run_league(info, objectives, algorithms, plot=False, log_all=False, **kwargs):
    results = []
    for season_id in info.seasons:
        for objective in objectives:
            for algorithm in algorithms:
                s, solve_info = realign(info.league, info.seasons[season_id], objective=objective, algorithm=algorithm, **kwargs)
                if log_all:
                    filename = f'{get_log_path()}/{info.name}_{season_id}_{objective}_{algorithm}.csv'
                    print('logging to ', filename)
                    log_table(filename, s)
                if plot:
                    plot_divisions(info.name, s)
                results.append([info.name, season_id, objective, algorithm, solve_info['objective'], solve_info['time'], get_arg(kwargs, 'solver', 'none')])
    df = pd.DataFrame(results, columns=['league', 'season', 'objective', 'algorithm', 'objective_value', 'time', 'solver'])
    log_table(get_log_filename(f'{info.name}'), df)
    return df

def run_all(objectives, algorithms, plot, filename='leagues', **kwargs):
    results = []
    for name in leagues:
        info = leagues[name]
        print(f"Running league {info.name} with {objectives} and {algorithms}")
        results.append(run_league(info, objectives, algorithms, plot, **kwargs))
    df = pd.concat(results)
    log_table(get_log_filename(filename), df)
    return df

def run_solvers(objectives, solvers=['gurobi', 'scip'], plot=False):
    results = []
    ls = [leagues['nfl-conf'], leagues['nba-conf'], leagues['nhl-conf'], leagues['mlb-conf']]
    for info in ls:
        print(f"Running league {info.name} with {objectives}")
        for solver in solvers:
            results.append(run_league(info, objectives, ['optimal'], plot, solver=solver))
    df = pd.concat(results)
    log_table(get_log_filename('solvers'), df)
    return df

def run_max_swaps(league_name='nfl-conf', season='2023-nfc', solver='gurobi'):
    results = []
    info = leagues[league_name]
    print(f"Running league {info.name}-{season} with max_swaps")
    for i in range(info.league.total_team_count()):
        s, solve_info = realign(info.league, info.seasons[season], max_swaps=i, solver=solver)
        results.append([info.name, 2023, i, solve_info['objective'], solve_info['time']])
    df = pd.DataFrame(results, columns=['league', 'season', 'max_swaps', 'objective_value', 'time'])
    log_table(get_log_filename('max_swaps'), df)
    return df

# to compress: convert mlb_inc.png -compress lzw eps2:mlb_inc.eps

def realignment_to_latex(df, caption='Realignment'):
    confs = df.groupby('conf')
    #divisions_in_conf = [len(conf.groupby('division')) for key, conf in confs]
    #print(divisions_in_conf)
    print("\\begin{table}")
    conf_div_count = [(key, len(conf.groupby('division'))) for key, conf in confs]
    tab = '|'.join(['l'*k for c, k in conf_div_count])
    print('\\begin{tabular}{' + tab + '}')
    print(' & '.join([f'\\multicolumn{{{n_d}}}{{l}}{{ {c} }}' for c, n_d in conf_div_count]) + ' \\\\')
    

    div_conf = [d.title() for key, conf in confs for d, div in conf.groupby('division')]
    print(' & '.join(div_conf) + ' \\\\')
    print('\\hline')

    t_by_cd = { c : {d: sorted([t for t in div['team_abbr']]) for d, div in conf.groupby('division')} for c, conf in confs}
    rows = max([max(map(len, t_by_cd[c].values())) for c, conf in confs])
    for i in range(rows):
        ts = [(t_by_cd[c][d][i] if i < len(t_by_cd[c][d]) else '') for c, conf in confs for d, div in conf.groupby('division')]
        print(' & '.join(ts) + ' \\\\')
    print('\\end{tabular}')
    print('\\caption{' + caption + '}')
    print('\\end{table}')

#r_all = pd.read_csv('out/leagues_2024_02_05.log')
def summary_to_latex(r_all, values, gaps, fmt='{:.2f}'):
    keys = ['league', 'season']
    algs = ['optimal', 'naive', 'greedy']
    r_piv = r_all.pivot(index=keys, columns='algorithm', values=values)
    format = {d:fmt for d in algs}
    cols = keys + algs
    if gaps:
        r_piv['greedy gap'] = (r_piv['greedy'] - r_piv['optimal']) / r_piv['optimal']
        r_piv['naive gap'] = (r_piv['naive'] - r_piv['optimal']) / r_piv['optimal']
        format.update({'greedy gap':'{:.2%}', 'naive gap':'{:.2%}'})
        cols += ['greedy', 'greedy gap']
    summary = r_piv.rename_axis(None, axis=1).reset_index()[cols]
    txt = summary.style.format(format).hide(level=0, axis=0).to_latex()
    print(txt)

def optimal_to_latex(league_name, season):
    df = pd.read_csv(get_path_optimal(league_name, season))
    realignment_to_latex(df)

def incumbent_to_latex(league_name, season):
    df = pd.read_csv(get_path_incumbent(league_name, season))
    realignment_to_latex(df)

def plot_incumbent(league_name, season=2023):
    df = leagues[league_name].seasons[season]
    fig = plot_divisions(league_name, leagues[league_name].seasons[season])
    fig.write_image(f'doc/{league_name}_{season}_inc.png')
    subprocess.run(['convert', f'doc/{league_name}_{season}_inc.png', '-compress', 'lzw', f'doc/{league_name}_{season}_inc.eps'])

def plot_optimal(league_name, season=2023):
    df = pd.read_csv(get_path_optimal(league_name, season))
    fig = plot_divisions(league_name, df)
    fig.write_image(f'doc/{league_name}_{season}_opt.png')
    subprocess.run(['convert', f'doc/{league_name}_{season}_opt.png', '-compress', 'lzw', f'doc/{league_name}_{season}_opt.eps'])

def stable_test(league='nfl-conf', season='2023-afc', step=1000):
    l = leagues[league]
    df = l.seasons[season]
    df_opt = pd.read_csv(get_path_optimal(league, season))
    obj_opt = score_distance(df_opt, make_distances(df_opt))
    lower = int(obj_opt / step) * step + step
    upper = int(score_distance(df, make_distances(df)) / step) * step
    rs = []

    sol, info = realign(l.league, df, objective='stability', algorithm='naive')
    print(info)
    rs.append([league, season, info['objective'], info['time'], 0])

    sol, info = realign(l.league, df, 
                       objective='stability', algorithm='optimal', relabel=True, solver='scip', 
                       linearize=True, d_max=obj_opt+1, verbose=True)
    last = info['team_swap_count']
    rs.append([league, season, info['objective'], info['time'], info['team_swap_count']])

    last = df.shape[0]
    for k in range(lower, upper + step, step):
        sol, info = realign(l.league, df, 
                       objective='stability', algorithm='optimal', relabel=True, solver='scip', 
                       linearize=True, d_max=k, verbose=True)
        if info['team_swap_count'] < last:
            rs.append([league, season, info['objective'], info['time'], info['team_swap_count']])
            last = info['team_swap_count']
    r = pd.DataFrame(rs, columns=['league', 'season', 'objective', 'time', 'team_swap_count'])
    c_obj = r['objective']
    r['optimality_gap'] = (c_obj.max() - c_obj) / c_obj
    return r.sort_values('optimality_gap')

# make an artificial league with conf_count conferences, div_count divisions, and total_team_count teams.
# Note that the number of teams in each division is not guaranteed to be the same.
def make_artificial_league(conf_count, div_count, total_team_count):
    confs = [f'c{i}' for i in range(conf_count)]
    divs = [f'd{i}' for i in range(div_count)]
    tc = {(c, d): total_team_count // (conf_count * div_count) for c in confs for d in divs}
    team_count = total_team_count // (conf_count * div_count)
    # allocate any remaining slots
    for i in range(total_team_count % (conf_count * div_count)):
        tc[confs[i // div_count], divs[i % div_count]] += 1
    league_data = [(c, d, tc[c, d]) for c in confs for d in divs]
    league = pd.DataFrame(league_data, columns=['conf', 'division', 'team_count'])
    team_data = [(f't{i}', f'c{i % conf_count}', f'd{i % div_count}', 0, 0) for i in range(total_team_count)]
    teams = pd.DataFrame(team_data, columns=['team_abbr', 'conf', 'division', 'team_lat', 'team_lng'])
    return League(league), teams

# Generate a problem instance from the test data provided by
# An Exact Approach for the Balanced k-Way Partitioning Problem with Weight Constraints and its Application to
# Sports Team Realignment. Diego Recalde, Daniel Sever´ın, Ramiro Torres, Polo Vaca
def read_recsev_instance(filename):
    with open(filename, 'r') as file:
        content = file.read()
    lines = content.split('\n')

    # Extract the size of the matrix and the number of clusters
    size = int(lines[0])
    num_divisions = int(lines[1])  
    print(num_divisions)

    # Extract the distance matrix
    data = np.array([list(map(float, row.split())) for row in lines[2:]][:size])
    # the first column are weights. the rest are distances
    league, df = make_artificial_league(1, num_divisions, size)
    teams = [t['team_abbr'] for i, t in df.iterrows()]
    distances = {(t, u): data[i, j+1] for i, t in enumerate(teams) for j, u in enumerate(teams)}
    return league, df, distances, data[:, 0]
