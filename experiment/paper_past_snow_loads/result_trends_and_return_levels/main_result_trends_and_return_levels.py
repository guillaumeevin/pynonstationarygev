from multiprocessing.pool import Pool

import matplotlib as mpl

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode
from experiment.paper_past_snow_loads.paper_main_utils import load_altitude_to_visualizer
from experiment.paper_past_snow_loads.paper_utils import paper_study_classes, paper_altitudes
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
    _ = visualizer.massif_name_to_minimized_aic_non_stationary_trend_test
    return True

def intermediate_result(altitudes, massif_names=None,
                        non_stationary_uncertainty=None, uncertainty_methods=None,
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
    altitude_to_visualizer = load_altitude_to_visualizer(altitudes, massif_names, non_stationary_uncertainty,
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
    altitudes_for_plot_trend = [900, 1800, 2700]
    visualizers_for_altitudes = [visualizer
                                 for altitude, visualizer in altitude_to_visualizer.items()
                                 if altitude in altitudes_for_plot_trend]
    max_abs_tdrl = max([visualizer.max_abs_change for visualizer in visualizers_for_altitudes])
    for visualizer in visualizers_for_altitudes:
        visualizer.plot_trends(max_abs_tdrl, add_colorbar=visualizer.study.altitude==2700)

    # Plot graph
    plot_uncertainty_massifs(altitude_to_visualizer)
    # Plot histogram
    plot_uncertainty_histogram(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][:]
    massif_names = None
    for study_class in paper_study_classes[:1]:
        if study_class == CrocusSnowLoadEurocode:
            non_stationary_uncertainty = [False]
        else:
            non_stationary_uncertainty = [False, True][:]
        intermediate_result(paper_altitudes, massif_names, non_stationary_uncertainty, uncertainty_methods, study_class)


if __name__ == '__main__':
    # major_result()
    # intermediate_result(altitudes=paper_altitudes, massif_names=['Maurienne'],
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                        ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     non_stationary_uncertainty=[False, True][1:],
    #                     multiprocessing=True)
    intermediate_result(altitudes=paper_altitudes, massif_names=['Maurienne'],
                        uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][:],
                        non_stationary_uncertainty=[False, True][:],
                        multiprocessing=True)
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
