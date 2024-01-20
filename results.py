import pandas as pd
from datetime import datetime
import timeit

def run(objective, algorithm):
    for info in leagues:
        print(f"League {info.name}")
        r = Realign(info.league, GurobiModel)
        for y in info.seasons:
            s = r.solve(info.seasons[y], objective=objective, algorithm=algorithm)
            plot_divisions(info.name, s)

# to compress: convert mlb_inc.png -compress lzw eps2:mlb_inc.eps