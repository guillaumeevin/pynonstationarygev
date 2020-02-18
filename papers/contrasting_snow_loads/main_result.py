from multiprocessing.pool import Pool

import matplotlib as mpl

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoad3Days, \
    CrocusSnowLoad5Days, CrocusSnowLoad7Days, CrocusSnowLoad1Day
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from papers.contrasting_snow_loads.plot_contrasting_trend_curves import plot_contrasting_trend_curves
from papers.exceeding_snow_loads.paper_main_utils import load_altitude_to_visualizer
from papers.exceeding_snow_loads.paper_utils import paper_study_classes, paper_altitudes
from papers.exceeding_snow_loads.result_trends_and_return_levels.main_result_trends_and_return_levels import \
    compute_minimized_aic
from papers.exceeding_snow_loads.result_trends_and_return_levels.plot_selection_curves import plot_selection_curves
from papers.exceeding_snow_loads.result_trends_and_return_levels.plot_trend_curves import plot_trend_curves, \
    plot_trend_map
from papers.exceeding_snow_loads.result_trends_and_return_levels.plot_uncertainty_curves import plot_uncertainty_massifs
from papers.exceeding_snow_loads.result_trends_and_return_levels.plot_uncertainty_histogram import \
    plot_uncertainty_histogram
from root_utils import NB_CORES

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def intermediate_result(altitudes, massif_names=None,
                        model_subsets_for_uncertainty=None, uncertainty_methods=None,
                        study_class=CrocusSnowLoad3Days,
                        multiprocessing=False):
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
                                                         study_class, uncertainty_methods)
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
    plot_contrasting_trend_curves(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    massif_names = None
    model_subsets_for_uncertainty = None
    altitudes = paper_altitudes
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700]
    study_classes = [CrocusSnowLoad1Day, CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days][:]
    for study_class in study_classes:
        intermediate_result(altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class)


if __name__ == '__main__':
    major_result()
    # intermediate_result(altitudes=[1500, 1800][:], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     multiprocessing=True)