from collections import OrderedDict

import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from projects.projected_snowfall.comparison_with_scm.comparison_plot import individual_plot, collective_plot

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import load_gcm_rcm_couples_for_year_min_and_year_max, \
    AdamontScenario
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def main():
    fast = True
    # Set the year_min and year_max for the comparison
    if fast is None:
        year_min = [1982, 1950][1]
        massif_names = ['Vanoise']
        altitudes = [1200, 1500, 1800, 2100, 2400]
    elif fast:
        year_min = [1982, 1950][0]
        massif_names = ['Vanoise']
        altitudes = [1800]
    else:
        massif_names = None
        year_min = [1982, 1950][1]
        altitudes = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]

    # Load visualizers
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        visualizer = load_visualizer(altitude, massif_names, year_min)
        altitude_to_visualizer[altitude] = visualizer
        # Individual plot
        individual_plot(visualizer)
    # Collective plot
    collective_plot(altitude_to_visualizer)


def load_visualizer(altitude, massif_names, year_min) -> ComparisonHistoricalVisualizer:
    year_min = max(1959, year_min)
    year_max = 2019
    study_class_couple = [(SafranSnowfall1Day, AdamontSnowfall)][0]
    scm_study_class, adamont_study_class = study_class_couple
    season = Season.annual
    adamont_scenario = AdamontScenario.rcp85_extended

    # Loading part
    scm_study = scm_study_class(altitude=altitude, year_min=year_min, year_max=year_max, season=season)
    gcm_rcm_couples = load_gcm_rcm_couples_for_year_min_and_year_max(year_min, year_max,
                                                                     adamont_scenario=adamont_scenario)
    adamont_studies = AdamontStudies(adamont_study_class, gcm_rcm_couples,
                                     altitude=altitude, year_min=year_min, year_max=year_max, season=season,
                                     scenario=adamont_scenario)
    visualizer = ComparisonHistoricalVisualizer(scm_study, adamont_studies, massif_names=massif_names)
    return visualizer


if __name__ == '__main__':
    main()
