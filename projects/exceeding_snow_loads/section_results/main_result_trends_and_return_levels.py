from multiprocessing.pool import Pool

import matplotlib as mpl

from projects.exceeding_snow_loads.checks.qqplot.plot_qqplot import \
    plot_intensity_against_gumbel_quantile_for_3_examples, plot_full_diagnostic
from projects.exceeding_snow_loads.section_results.plot_selection_curves import plot_selection_curves
from projects.exceeding_snow_loads.section_results.plot_trend_curves import plot_trend_map, plot_trend_curves
from projects.exceeding_snow_loads.section_results.plot_uncertainty_histogram import plot_uncertainty_histogram

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends, ModelSubsetForUncertainty
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
    print(visualizer.altitude)
    _ = visualizer.massif_name_to_trend_test_that_minimized_aic
    return True

def intermediate_result(altitudes, massif_names=None,
                        model_subsets_for_uncertainty=None, uncertainty_methods=None,
                        study_class=CrocusSnowLoadTotal,
                        multiprocessing=False,
                        only_histogram=False):
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
    if only_histogram:
        plot_uncertainty_histogram(altitude_to_visualizer)
    else:
        # Load variable object efficiently
        for v in altitude_to_visualizer.values():
            _ = v.study.year_to_variable_object
        # Compute minimized value efficiently
        # visualizers = list()
        if multiprocessing:
            with Pool(NB_CORES) as p:
                _ = p.imap(compute_minimized_aic, altitude_to_visualizer.values())
        else:
            for visualizer in altitude_to_visualizer.values():
                _ = compute_minimized_aic(visualizer)

        # Plots
        # plot_trend_map(altitude_to_visualizer)
        # plot_trend_curves(altitude_to_visualizer={a: v for a, v in altitude_to_visualizer.items() if a >= 900})
        plot_uncertainty_massifs(altitude_to_visualizer)
        plot_uncertainty_histogram(altitude_to_visualizer)
        # plot_selection_curves(altitude_to_visualizer)
        # plot_intensity_against_gumbel_quantile_for_3_examples(altitude_to_visualizer)

        # Additional plots
        # uncertainty_interval_size(altitude_to_visualizer)
        # plot_full_diagnostic(altitude_to_visualizer)


def major_result():
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    # massif_names = ['Beaufortain', 'Vercors']
    massif_names = None
    study_classes = paper_study_classes[:]
    # study_classes = [CrocusSnowLoad3Days, CrocusSnowLoad5Days, CrocusSnowLoad7Days][::-1]
    altitudes = [300, 600, 900, 1800, 2700][:2]
    altitudes = [300, 600, 900, 1200, 1500, 1800]
    # altitudes = paper_altitudes
    # altitudes = [900, 1800, 270{{0][:1]
    for study_class in study_classes:
        print('new stuy class', study_class)
        if study_class is CrocusSnowLoadEurocode:
            model_subsets_for_uncertainty = [ModelSubsetForUncertainty.stationary_gumbel]
            only_histogram = True
        else:
            model_subsets_for_uncertainty = None
            only_histogram = False
        intermediate_result(altitudes, massif_names, model_subsets_for_uncertainty,
                            uncertainty_methods, study_class,
                            multiprocessing=True,
                            only_histogram=only_histogram)


if __name__ == '__main__':
    major_result()
    # intermediate_result(altitudes=[300], massif_names=None,
    #                     uncertainty_methods=[ConfidenceIntervalMethodFromExtremes.my_bayes,
    #                                          ConfidenceIntervalMethodFromExtremes.ci_mle][1:],
    #                     multiprocessing=True)
