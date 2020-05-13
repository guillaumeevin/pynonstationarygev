from multiprocessing.pool import Pool

import matplotlib as mpl

from projects.exceeding_snow_loads.section_results.plot_selection_curves import plot_selection_curves
from projects.exceeding_snow_loads.section_results.plot_trend_curves import plot_trend_map

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_trend.visualizers.utils import load_altitude_to_visualizer
from projects.exceeding_snow_loads.section_results.plot_uncertainty_curves import plot_uncertainty_massifs
from projects.exceeding_snow_loads.utils import paper_study_classes, paper_altitudes
from root_utils import NB_CORES


import matplotlib.pyplot as plt


def minor_result(altitude):
    """Plot trends for a single altitude to be fast"""
    visualizer = StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude), multiprocessing=True,
                                                       )
    visualizer.plot_trends()
    plt.show()


def compute_minimized_aic(visualizer):
    _ = visualizer.massif_name_to_trend_test_that_minimized_aic
    return True


def intermediate_result(altitudes, massif_names=None,
                        model_subsets_for_uncertainty=None, uncertainty_methods=None,
                        study_class=CrocusSnowLoadTotal,
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
    # plot_trend_map(altitude_to_visualizer)
    # plot_trend_curves(altitude_to_visualizer={a: v for a, v in altitude_to_visualizer.items() if a >= 900})
    # plot_uncertainty_massifs(altitude_to_visualizer)
    # plot_uncertainty_histogram(altitude_to_visualizer)
    plot_selection_curves(altitude_to_visualizer)
    # uncertainty_interval_size(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    # massif_names = ['Beaufortain', 'Vercors']
    massif_names = None
    study_classes = paper_study_classes[:1]
    # model_subsets_for_uncertainty = [ModelSubsetForUncertainty.stationary_gumbel,
    #                                  ModelSubsetForUncertainty.stationary_gumbel_and_gev,
    #                                  ModelSubsetForUncertainty.non_stationary_gumbel,
    #                                  ModelSubsetForUncertainty.non_stationary_gumbel_and_gev]
    model_subsets_for_uncertainty = None
    # study_classes = [CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days][::-1]
    altitudes = paper_altitudes
    # altitudes = [900]
    for study_class in study_classes:
        intermediate_result(altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class,
                            multiprocessing=True)


if __name__ == '__main__':
    major_result()
    # intermediate_result(altitudes=[300], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     multiprocessing=True)
