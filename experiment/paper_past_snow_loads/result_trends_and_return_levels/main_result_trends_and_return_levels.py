from collections import OrderedDict
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.paper_past_snow_loads.result_trends_and_return_levels.eurocode_visualizer import \
    plot_uncertainty_massifs
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.paper_past_snow_loads.result_trends_and_return_levels.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from root_utils import VERSION_TIME

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def minor_result(altitude):
    """Plot trends for a single altitude to be fast"""
    visualizer = StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude), multiprocessing=True)
    visualizer.plot_trends()


def intermediate_result(altitudes, massif_names=None,
                        non_stationary_uncertainty=None, uncertainty_methods=None,
                        study_class=CrocusSnowLoadTotal):
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
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        altitude_to_visualizer[altitude] = StudyVisualizerForNonStationaryTrends(
            study=study_class(altitude=altitude), multiprocessing=True, save_to_file=True,
            uncertainty_massif_names=massif_names, uncertainty_methods=uncertainty_methods,
            non_stationary_contexts=non_stationary_uncertainty)
    # Plot trends
    max_abs_tdrl = max([visualizer.max_abs_tdrl for visualizer in altitude_to_visualizer.values()])
    for visualizer in altitude_to_visualizer.values():
        visualizer.plot_trends(max_abs_tdrl)
    # Plot graph
    plot_uncertainty_massifs(altitude_to_visualizer)
    return altitude_to_visualizer


def major_result():
    altitudes = [[1500, 1800]][0]
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    massif_names = ['Chartreuse']
    non_stationary_models_for_uncertainty = [False, True][:1]
    #
    # altitudes
    # study_class


if __name__ == '__main__':
    # minor_result(altitude=1800)
    # intermediate_result(altitudes=[1500, 1800], massif_names=['Chartreuse'],
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False])
    # intermediate_result(altitudes=[1500, 1800], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False])
    # intermediate_result(altitudes=[300, 600, 900, 1200, 1500, 1800], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle],
    #                     non_stationary_uncertainty=[False])
    intermediate_result(altitudes=[300, 600, 900, 1200, 1500, 1800], massif_names=None,
                        uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.ci_mle,
                                             ConfidenceIntervalMethodFromExtremes.ci_bayes],
                        non_stationary_uncertainty=[False, True])
