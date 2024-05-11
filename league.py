from datetime import datetime
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

    def total_team_count(self):
        return sum([d['team_count'] for d in self.all_divisions.values()])


    @staticmethod
    def read_csv(filename):
        df = pd.read_csv(filename)
        return League(df)

def get_arg(args, key, default):
    if key in args:
        print(f'Using {key}={args[key]}')
        return args[key]
    else:
        return default
    

def read_season_data(teams):
    df = pd.read_csv(teams)
    df['team'] = df.index
    return df


def read_ratings_data(ratings):
    df = pd.read_csv(ratings)
    return df

def get_log_path():
    return 'log'

def get_log_filename(base):
    suffix = datetime.now().strftime("%Y_%m_%d")
    return f'{get_log_path()}/{base}_{suffix}.log'


def log_table(logfile, result):
    result.to_csv(logfile, index=False,  mode='a')

def log_to_file(logfile, msg):
    with open(logfile, 'a') as f:
        f.write(f'{msg}\n')

nfl = League.read_csv('data/nfl.csv')
nfl_data = { 2002 : read_season_data("data/nfl-2002.csv"),
             2023 : read_season_data("data/nfl-2023.csv"),
           }

nfl_conf = League.read_csv('data/nfl-conf.csv')
nfl_conf_data = { 
             '2023-nfc' : read_season_data("data/nfl-2023-nfc.csv"),
             '2023-afc' : read_season_data("data/nfl-2023-afc.csv"),
           }

nhl = League.read_csv('data/nhl.csv')
nhl_data = { 
             2023 : read_season_data("data/nhl-2023.csv")
           }

nhl_conf = League.read_csv('data/nhl-conf.csv')
nhl_conf_data = { 
             '2023-east' : read_season_data("data/nhl-2023-east.csv"),
             '2023-west' : read_season_data("data/nhl-2023-west.csv"),
           }

mlb = League.read_csv('data/mlb.csv')
mlb_data = { 
             2023 : read_season_data("data/mlb-2023.csv")
           }

mlb_conf = League.read_csv('data/mlb-conf.csv')
mlb_conf_data = { 
             '2023-al' : read_season_data("data/mlb-2023-al.csv"),
             '2023-nl' : read_season_data("data/mlb-2023-nl.csv"),
           }

nba = League.read_csv('data/nba.csv')
nba_data = { 
             2023 : read_season_data("data/nba-2023.csv")
           }

nba_conf = League.read_csv('data/nba-conf.csv')
nba_conf_data = { 
             '2023-east' : read_season_data("data/nba-2023-east.csv"),
             '2023-west' : read_season_data("data/nba-2023-west.csv"),
           }

class LeagueInfo:
    def __init__(self, name, league, seasons):
        self.name = name
        self.league = league
        self.seasons = seasons

leagues = {
    'nfl' : LeagueInfo('nfl', nfl, nfl_data),
    'nfl-conf' : LeagueInfo('nfl-conf', nfl_conf, nfl_conf_data),
    'mlb' : LeagueInfo('mlb', mlb, mlb_data),
    'mlb-conf' : LeagueInfo('mlb-conf', mlb_conf, mlb_conf_data),
    'nba' : LeagueInfo('nba', nba, nba_data),
    'nba-conf' : LeagueInfo('nba-conf', nba_conf, nba_conf_data),
    'nhl' : LeagueInfo('nhl', nhl, nhl_data),
    'nhl-conf' : LeagueInfo('nhl-conf', nhl_conf, nhl_conf_data)
}
