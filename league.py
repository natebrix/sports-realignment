import pandas as pd

class League:
    def __init__(self, df):
        self.all_divisions = df.set_index(['conf', 'division']).to_dict(orient='index')
        self.confs = df['conf'].unique()
        self.div_by_conf = {k:list(g['division']) for (k, g) in df.groupby('conf')}

    def divisions(self, c):
        return self.div_by_conf[c]
    
    def team_count(self, c, d):
        return self.all_divisions[c, d]['team_count']

    @staticmethod
    def read_csv(filename):
        df = pd.read_csv(filename)
        return League(df)

def get_locations(teams):
    df = pd.read_csv(teams)
    df['team'] = df.index
    return df


def get_ratings(ratings):
    df = pd.read_csv(ratings)
    return df

nfl = League.read_csv('data/nfl.csv')
nfl_data = { 2002 : get_locations("data/nfl-2002.csv"),
             2023 : get_locations("data/nfl-2023.csv"),
             'ratings-2023': get_ratings("data/nfl-2023-rating.csv")
           }

nhl = League.read_csv('data/nhl.csv')
nhl_data = { 
             2023 : get_locations("data/nhl-2023.csv")
           }

mlb = League.read_csv('data/mlb.csv')
mlb_data = { 
             2023 : get_locations("data/mlb-2023.csv")
           }

nba = League.read_csv('data/nba.csv')
nba_data = { 
             2023 : get_locations("data/nba-2023.csv")
           }

