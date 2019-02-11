import pandas as pd

from extreme_estimator.gev_params import GevParams
from safran_study.safran import Safran
from safran_study.safran_extended import ExtendedSafran
from utils import VERSION
from itertools import product


def load_all_safran(only_first_one=False):
    all_safran = []
    for safran_alti, nb_day in product([1800, 2400], [1, 3, 7]):
        print('alti: {}, nb_day: {}'.format(safran_alti, nb_day))
        all_safran.append(Safran(safran_alti, nb_day))
        if only_first_one:
            break
    return all_safran


def fit_mle_gev_independent():
    # Dump the result in a csv
    dfs = []
    for safran in load_all_safran(only_first_one=True):
        safran.visualize_gev_fit_with_cmap()
        # path = r'/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/results/fit_mle_massif/fit_mle_gev_{}.csv'
        # df.to_csv(path.format(VERSION))


def fit_max_stab():
    pass


if __name__ == '__main__':
    fit_mle_gev_independent()
