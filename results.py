from datetime import datetime
import timeit
import pandas as pd

# mahalanobis - TV and distance
# plot times by size

def write_by_conference(league, season):
    df = league.seasons[season]
    g = df.groupby('conf')
    for conf in g.groups:
        df_c = g.get_group(conf)
        filename = f'{league.name.lower()}-{season}-{conf.lower()}.csv' 
        print(f"Writing {filename}")
        df_c.to_csv(f'data/{filename}', index=False)

def run_league(info, objectives, algorithms, plot, **kwargs):
    results = []
    for y in info.seasons:
        for objective in objectives:
            for algorithm in algorithms:
                s, obj, t = realign(info.league, info.seasons[y], objective=objective, algorithm=algorithm, **kwargs)
                if plot:
                    plot_divisions(info.name, s)
                results.append([info.name, y, objective, algorithm, obj, t, get_arg(kwargs, 'solver', 'none')])
    return pd.DataFrame(results, columns=['league', 'season', 'objective', 'algorithm', 'objective_value', 'time', 'solver'])

def run(objectives, algorithms, plot, **kwargs):
    results = []
    for name in leagues:
        info = leagues[name]
        print(f"Running league {info.name} with {objectives} and {algorithms}")
        results.append(run_league(info, objectives, algorithms, plot, **kwargs))
    df = pd.concat(results)
    # log_to_file(r.logfile, df.to_csv(index=False)) todo need logger after all
    return df

def run_solvers(objectives, solvers=['gurobi', 'scip'], plot=False):
    results = []
    ls = [leagues['nfl-conf'], leagues['nba-conf'], leagues['nhl-conf'], leagues['mlb-conf']]
    for info in ls:
        print(f"Running league {info.name} with {objectives}")
        for solver in solvers:
            results.append(run_league(info, objectives, ['optimal'], plot, solver=solver))
    df = pd.concat(results)
    #log_to_file(r.logfile, df.to_csv(index=False))
    return df

# todo fix me. Debug these results. I am not sure they are right.
def run_max_swaps(league_name='nfl-conf', season='2023-nfc'):
    results = []
    info = leagues[league_name]
    print(f"Running league {info.name}-{season} with max_swaps")
    for i in range(info.league.total_team_count()):
        s, obj, t = realign(info.league, info.seasons[season], max_swaps=i, solver='gurobi')
        results.append([info.name, 2023, i, obj, t])
    df = pd.DataFrame(results, columns=['league', 'season', 'max_swaps', 'objective_value', 'time'])
    #log_to_file(r.logfile, df.to_csv(index=False))
    return df

# to compress: convert mlb_inc.png -compress lzw eps2:mlb_inc.eps