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
