import pandas as pd

from safran_study.abstract_safran import Safran
from utils import VERSION


class Safran1800(Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safran_altitude = 1800


class Safran2400(Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safran_altitude = 2400

def fit_mle_gev_for_all_safran_and_different_days():
    # Dump the result in a csv
    dfs = []
    for safran_class in [Safran1800, Safran2400]:
        for nb_day in [1, 3, 7]:
            print('here')
            safran = safran_class(nb_day)
            df = safran.df_gev_mle_each_massif
            df.index += ' Safran{} with {} days'.format(safran.safran_altitude, safran.nb_days_of_snowfall)
            dfs.append(df)
    df = pd.concat(dfs)
    path = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/results/fit_mle_massif/fit_mle_gev_{}.csv'
    df.to_csv(path.format(VERSION))


if __name__ == '__main__':
    fit_mle_gev_for_all_safran_and_different_days()