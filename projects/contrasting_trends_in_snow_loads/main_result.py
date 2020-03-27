from multiprocessing.pool import Pool

import matplotlib as mpl

from extreme_data.meteo_france_data.scm_models_data.utils import SeasonForTheMaxima
from extreme_trend.visualizers.utils import load_altitude_to_visualizer

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation3Days, \
    SafranPrecipitation1Day, SafranPrecipitation5Days, SafranPrecipitation7Days, SafranSnowfall1Day, \
    SafranSnowfall5Days, SafranSnowfall3Days, SafranSnowfall7Days, SafranRainfall1Day, SafranRainfall3Days, \
    SafranRainfall5Days, SafranRainfall7Days

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, \
    CrocusSnowLoad5Days, CrocusSnowLoad7Days, CrocusSnowLoad1Day
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from projects.contrasting_trends_in_snow_loads.plot_contrasting_trend_curves import plot_contrasting_trend_curves
from projects.exceeding_snow_loads.section_results.main_result_trends_and_return_levels import \
    compute_minimized_aic
from root_utils import NB_CORES


def intermediate_result(altitudes, massif_names=None,
                        model_subsets_for_uncertainty=None, uncertainty_methods=None,
                        study_class=AbstractStudy,
                        multiprocessing=False,
                        save_to_file=True):
    """
    Plot all the trends for all altitudes
    And enable to plot uncertainty plot for some specific massif_names, uncertainty methods to be fast
    :param altitudes:
    :param massif_names:
    :param non_stationary_uncertainty:
    :param uncertainty_methods:
    :param study_class:
    :return:
    """
    # Load altitude to visualizer
    altitude_to_visualizer = load_altitude_to_visualizer(altitudes, massif_names, model_subsets_for_uncertainty,
                                                         study_class, uncertainty_methods, save_to_file=save_to_file)
    # Load variable object efficiently
    for v in altitude_to_visualizer.values():
        _ = v.study.year_to_variable_object
    # Compute minimized value efficiently
    visualizers = list(altitude_to_visualizer.values())
    if multiprocessing:
        with Pool(NB_CORES) as p:
            _ = p.map(compute_minimized_aic, visualizers)
    else:
        for visualizer in visualizers:
            _ = compute_minimized_aic(visualizer)

    # Plots
    plot_contrasting_trend_curves(altitude_to_visualizer, all_regions=True)

def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    massif_names = None
    model_subsets_for_uncertainty = None
    # altitudes = paper_altitudes
    # altitudes = paper_altitudes
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][:]
    snow_load_classes = [CrocusSnowLoad1Day, CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days][:]
    precipitation_classes = [SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days,
                             SafranPrecipitation7Days][:]
    snowfall_classes = [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days]
    rainfall_classes = [SafranRainfall1Day, SafranRainfall3Days, SafranRainfall5Days, SafranRainfall7Days]
    study_classes = precipitation_classes + snow_load_classes
    # study_classes = snowfall_classes + rainfall_classes
    for study_class in snowfall_classes:
        intermediate_result(altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class, multiprocessing=True)

"""

est ce qu il y a une croissance signifcative en pluie, 

est ce qu'il y a une decroissance signifcatieve à partir d'une certaine altitude
"""

if __name__ == '__main__':
    major_result()
    # intermediate_result(altitudes=[1500, 1800][:], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     multiprocessing=True,
    #                     save_to_file=False)
