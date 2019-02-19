import pandas as pd

from experiment.meteo_france_SCM_study.safran.safran import ExtendedSafran
from utils import VERSION


def fit_mle_gev_for_all_safran_and_different_days():
    # Dump the result in a csv
    dfs = []
    for safran_alti in [1800, 2400][:1]:
        for nb_day in [1, 3, 7][:]:
            print('alti: {}, nb_day: {}'.format(safran_alti, nb_day))
            # safran = Safran(safran_alti, nb_day)
            safran = ExtendedSafran(safran_alti, nb_day)
            df = safran.df_gev_mle_each_massif
            df.index += ' Safran{} with {} days'.format(safran.altitude, safran.nb_days_of_snowfall)
            dfs.append(df)
    df = pd.concat(dfs)
    path = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/results/fit_mle_massif/fit_mle_gev_{}.csv'
    df.to_csv(path.format(VERSION))


if __name__ == '__main__':
    fit_mle_gev_for_all_safran_and_different_days()
