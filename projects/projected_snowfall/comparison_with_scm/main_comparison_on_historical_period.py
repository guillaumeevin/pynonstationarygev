import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_full_name, \
    get_year_min_and_year_max_from_scenario, AdamontScenario, load_gcm_rcm_couples_for_year_min_and_year_max
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from projects.projected_snowfall.comparison_with_scm.comparison_historical_visualizer import \
    ComparisonHistoricalVisualizer


def main():
    fast = True
    # Set the year_min and year_max for the comparison
    if fast:
        year_min = [1982, 1950][1]
        massif_names = ['Vanoise']
        altitudes = [1800]
    else:
        massif_names = None
        year_min = [1982, 1950][1]
        altitudes = [900, 1800, 2700]

    for altitude in altitudes:
        plot(altitude, massif_names, year_min)


def plot(altitude, massif_names, year_min):
    year_min = max(1959, year_min)
    year_max = 2005
    study_class_couple = [(SafranSnowfall1Day, AdamontSnowfall)][0]
    scm_study_class, adamont_study_class = study_class_couple
    season = Season.annual
    scm_study = scm_study_class(altitude=altitude, year_min=year_min, year_max=year_max, season=season)
    gcm_rcm_couples = load_gcm_rcm_couples_for_year_min_and_year_max(year_min, year_max)
    adamont_studies = AdamontStudies(adamont_study_class, gcm_rcm_couples,
                                     altitude=altitude, year_min=year_min, year_max=year_max, season=season)
    adamont_studies.plot_maxima_time_series(massif_names, scm_study)
    visualizer = ComparisonHistoricalVisualizer(scm_study, adamont_studies, massif_names=massif_names)
    for plot_maxima in [True, False][:1]:
        visualizer.plot_comparison(plot_maxima)


if __name__ == '__main__':
    main()