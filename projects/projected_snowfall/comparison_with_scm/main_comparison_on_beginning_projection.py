

import matplotlib as mpl

from projects.projected_snowfall.comparison_with_scm.main_comparison_on_historical_period import load_vizualizer_histo

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
from collections import OrderedDict

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def main():
    fast = True
    # Set the year_min and year_max for the comparison
    if fast:
        year_min = [2006][0]
        year_max = [2019][0]
        massif_names = ['Vanoise']
        altitudes = [1800]
    else:
        year_min = [2006][0]
        year_max = [2019][0]
        massif_names = None
        altitudes = [900, 1800, 2700]

    # Load visualizers
    for altitude in altitudes:
        study_class_couple = [(SafranSnowfall1Day, AdamontSnowfall)][0]
        v = load_vizualizer_histo(altitude, massif_names, study_class_couple, year_max, year_min)
        v.adamont_studies.plot_maxima_time_series(v.massif_names, v.scm_study)


if __name__ == '__main__':
    main()
