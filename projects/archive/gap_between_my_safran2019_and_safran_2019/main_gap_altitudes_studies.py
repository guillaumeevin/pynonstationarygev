import datetime
import time
from typing import List

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.safran.gap_between_study import GapBetweenSafranSnowfall2019And2020, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies

matplotlib.use('Agg')

from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020

import matplotlib as mpl

from extreme_fit.model.utils import set_seed_for_test

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    study_classes = [GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered,
                     GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered,
                     GapBetweenSafranSnowfall2019And2020, SafranSnowfall2020, SafranSnowfall1Day, SafranSnowfall3Days,
                     SafranSnowfall5Days, SafranSnowfall7Days][1:2]
    seasons = [Season.annual, Season.winter, Season.spring, Season.automn][:1]

    set_seed_for_test()

    fast = True
    if fast:
        altitudes_list = altitudes_for_groups[:1]
    else:
        altitudes_list = altitudes_for_groups[1:]

    start = time.time()
    main_loop_time_series_only(altitudes_list, seasons, study_classes)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def main_loop_time_series_only(altitudes_list, seasons, study_classes):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for season in seasons:
        for study_class in study_classes:
            print('Inner loop', season, study_class)
            for altitudes in altitudes_list:
                studies = AltitudesStudies(study_class, altitudes, season=season)
                studies.plot_maxima_time_series()

if __name__ == '__main__':
    main()
