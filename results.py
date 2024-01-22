import pandas as pd
from datetime import datetime
import timeit

def run(objectives, algorithms, plot=False):
    results = []
    for info in leagues:
        print(f"Running league {info.name} with {objectives} and {algorithms}")
        r = Realign(info.league, GurobiModel)
        for y in info.seasons:
            for objective in objectives:
                for algorithm in algorithms:
                    obj, s = r.solve(info.seasons[y], objective=objective, algorithm=algorithm)
                    if plot:
                        plot_divisions(info.name, s)
                    results.append([info.name, y, objective, algorithm, obj])
    df = pd.DataFrame(results, columns=['league', 'season', 'objective', 'algorithm', 'objective_value'])
    log_to_file(r.logfile, df.to_csv(index=False))
    return df

# to compress: convert mlb_inc.png -compress lzw eps2:mlb_inc.eps