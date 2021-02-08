

import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
from collections import OrderedDict

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name, AdamontScenario, \
    load_gcm_rcm_couples_for_year_min_and_year_max
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def main():
    fast = True
    adamont_scenario = [AdamontScenario.histo, AdamontScenario.rcp85_extended][0]
    year_min = 1982 if adamont_scenario is AdamontScenario.rcp85_extended else 2006
    # Set the year_min and year_max for the comparison
    if fast is None:
        year_max = [2030][0]
        massif_names = ['Vanoise']
        altitudes = [1800]
    elif fast is None:
        # year_min = [1951][0]
        # year_min = [1951][0]
        year_max = [2005][0]
        massif_names = ['Vercors']
        altitudes = [900, 1200, 1500, 1800, 2100, 2400][3:4]
    else:
        year_max = [2100][0]
        massif_names = None
        altitudes = [900, 1800, 2700, 3600][2:]

    # Load studies
    for altitude in altitudes:
        adamont_study_class = AdamontSnowfall
        season = Season.annual
        gcm_rcm_couples = load_gcm_rcm_couples_for_year_min_and_year_max(year_min, year_max,
                                                                         adamont_scenario=adamont_scenario)
        adamont_studies = AdamontStudies(adamont_study_class, gcm_rcm_couples,
                                         altitude=altitude, year_min=year_min,
                                         year_max=year_max, season=season,
                                         scenario=adamont_scenario,
                                         adamont_version=2)
        adamont_studies.plot_maxima_time_series_adamont(massif_names)


if __name__ == '__main__':
    main()
