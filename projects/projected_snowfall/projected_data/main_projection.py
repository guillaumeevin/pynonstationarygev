

import matplotlib as mpl

from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
from collections import OrderedDict

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, \
    load_gcm_rcm_couples
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def main():
    fast = None
    adamont_scenario = [AdamontScenario.histo, AdamontScenario.rcp85_extended][1]
    year_min = 1982 if adamont_scenario is AdamontScenario.rcp85_extended else 2006
    # Set the year_min and year_max for the comparison
    if fast is True:
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
        gcm_rcm_couples = load_gcm_rcm_couples(year_min, year_max,
                                               adamont_scenario=adamont_scenario)
        adamont_studies = AdamontStudies(adamont_study_class, gcm_rcm_couples,
                                         altitude=altitude, year_min=year_min,
                                         year_max=year_max, season=season,
                                         scenario=adamont_scenario,
                                         adamont_version=2)
        if year_max <= 2020:
            scm_study = SafranSnowfall2020(altitude=altitude, season=season, year_min=year_min, year_max=year_max)
        else:
            scm_study = None
        adamont_studies.plot_maxima_time_series_adamont(massif_names, scm_study=scm_study)


if __name__ == '__main__':
    main()
