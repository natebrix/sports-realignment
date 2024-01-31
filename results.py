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

def run_league(info, objectives, algorithms, model, plot):
    results = []
    r = Realign(info.league, model)
    for y in info.seasons:
        for objective in objectives:
            for algorithm in algorithms:
                s, obj, t = r.solve(info.seasons[y], objective=objective, algorithm=algorithm)
                if plot:
                    plot_divisions(info.name, s)
                results.append([info.name, y, objective, algorithm, obj, t, model.solver_name])
    return pd.DataFrame(results, columns=['league', 'season', 'objective', 'algorithm', 'objective_value', 'time', 'solver'])

def run(objectives, algorithms, plot=False, model=GurobiModel):
    results = []
    for name in leagues:
        info = leagues[name]
        print(f"Running league {info.name} with {objectives} and {algorithms}")
        results.append(run_league(info, objectives, algorithms, model, plot))
    df = pd.concat(results)
    log_to_file(r.logfile, df.to_csv(index=False))
    return df

def run_solvers(objectives, models=[GurobiModel, ScipModel], plot=False):
    results = []
    ls = [leagues['nfl-conf'], leagues['nba-conf'], leagues['nhl-conf'], leagues['mlb-conf']]
    for info in ls:
        print(f"Running league {info.name} with {objectives}")
        for model in models:
            results.append(run_league(info, objectives, ['optimal'], model, plot))
    df = pd.concat(results)
    log_to_file(r.logfile, df.to_csv(index=False))
    return df

# to compress: convert mlb_inc.png -compress lzw eps2:mlb_inc.eps