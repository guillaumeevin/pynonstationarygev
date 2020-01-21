from multiprocessing.pool import Pool

import matplotlib as mpl

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from experiment.paper_past_snow_loads.paper_main_utils import load_altitude_to_visualizer
from experiment.paper_past_snow_loads.paper_utils import paper_study_classes, paper_altitudes, ModelSubsetForUncertainty
from experiment.paper_past_snow_loads.result_trends_and_return_levels.plot_trend_curves import plot_trend_curves
from experiment.paper_past_snow_loads.result_trends_and_return_levels.plot_uncertainty_curves import \
    plot_uncertainty_massifs
from experiment.paper_past_snow_loads.result_trends_and_return_levels.plot_uncertainty_histogram import \
    plot_uncertainty_histogram
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from root_utils import NB_CORES

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
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
    # Compute common max value for the colorbar
    max_abs_tdrl = max([visualizer.max_abs_change for altitude, visualizer in altitude_to_visualizer.items()
                        if altitude >= 900])
    for altitude, visualizer in altitude_to_visualizer.items():
        if 900 <= altitude <= 4200:
            add_color = (visualizer.study.altitude - 1500) % 900 == 0
            visualizer.plot_trends(max_abs_tdrl, add_colorbar=add_color)
            # Plot 2700 also with a colorbar
            if altitude == 2700:
                visualizer.plot_trends(max_abs_tdrl, add_colorbar=True)
        else:
            visualizer.plot_trends(None, add_colorbar=True)

    # Plot trends
    altitude_to_visualizer_for_plot_trend = {a: v for a, v in altitude_to_visualizer.items() if a >= 900}
    plot_trend_curves(altitude_to_visualizer_for_plot_trend)
    # Plot graph
    # plot_uncertainty_massifs(altitude_to_visualizer)
    # # Plot histogram
    # plot_uncertainty_histogram(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    massif_names = None
    study_classes = paper_study_classes[:2]
    # model_subsets_for_uncertainty = [ModelSubsetForUncertainty.stationary_gumbel,
    #                                  ModelSubsetForUncertainty.stationary_gumbel_and_gev,
    #                                  ModelSubsetForUncertainty.non_stationary_gumbel,
    #                                  ModelSubsetForUncertainty.non_stationary_gumbel_and_gev]
    model_subsets_for_uncertainty = None
    # study_classes = [CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days][::-1]
    for study_class in study_classes:
        intermediate_result(paper_altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class)


if __name__ == '__main__':
    # major_result()
    intermediate_result(altitudes=ALL_ALTITUDES_WITHOUT_NAN[2:], massif_names=None,
                        uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
                                             ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
                        multiprocessing=True)
    # intermediate_result(altitudes=[900, 1200], massif_names=['Maurienne'],
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     non_stationary_uncertainty=[False, True][:],
    #                     multiprocessing=True)
    # intermediate_result(altitudes=[900, 1200], massif_names=None)
    # intermediate_result(ALL_ALTITUDES_WITHOUT_NAN)
    # intermediate_result(paper_altitudes)
    # minor_result(altitude=900)
    # intermediate_result(altitudes=[1500, 1800], massif_names=['Chartreuse'],
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_bayes],
    #                     non_stationary_uncertainty=[True])
    # intermediate_result(altitudes=[1500, 1800], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False])
    # intermediate_result(altitudes=[300, 600, 900, 1200, 1500, 1800], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False])
    # intermediate_result(altitudes=[300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_bayes],
    #                     non_stationary_uncertainty=[False, True])
    # intermediate_result(altitudes=[300, 600, 900], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False, True])