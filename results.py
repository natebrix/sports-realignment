import pandas as pd
from datetime import datetime
import timeit

def run(objective):
    for info in leagues:
        print(f"League {info.name}")
        r = Realign(info.league, GurobiModel)
        for y in info.seasons:
            s = r.solve(info.seasons[y], objective=objective)
            plot_divisions(info.name, s)