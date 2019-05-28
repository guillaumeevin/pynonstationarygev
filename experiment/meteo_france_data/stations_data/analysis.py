from collections import OrderedDict

DATA_PATH = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/Johan_data/PrecipitationsAvalanches_MaxPrecipit_ParPoste_ParHiver_traites.xls'

import pandas as pd


class Stations(object):

    def load_main_df(self):
        df = pd.read_excel(DATA_PATH, sheet_name='max alpes 2500m presentes')
        df = df.iloc[:78, 4:]
        return df

    def reduce_altitude(self, altitude=900) -> pd.Series:
        df = self.load_main_df()
        ind_altitude = altitude - 150 < df['ALTITUDE']
        ind_altitude &= df['ALTITUDE'] <= altitude + 150
        df = df.loc[ind_altitude]
        # Put all the result into an ordered dict
        d = OrderedDict()
        # Number of stations
        d['Nb stations'] = len(df)
        # Number of massifs
        # d['Nb mas'] = len(df)
        return pd.Series(d)

    def df_altitude(self):
        altitudes = [900, 1200]
        df = pd.concat([self.reduce_altitude(altitude) for altitude in altitudes], axis=1)
        df = df.transpose()
        df.index = altitudes
        print(df)


if __name__ == '__main__':
    s = Stations()
    # s.load_main_df()
    s.df_altitude()
