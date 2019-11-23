from collections import OrderedDict
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.paper_past_snow_loads.result_trends_and_return_levels.eurocode_visualizer import \
    plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.paper_past_snow_loads.result_trends_and_return_levels.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from root_utils import VERSION_TIME

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def draw_snow_load_map(altitude):
    visualizer = StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude), multiprocessing=True)
    visualizer.plot_trends()


def main_results():
    altitudes = [[1500, 1800]][0]
    uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                           ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
    massif_names = ['Chartreuse']
    non_stationary_models_for_uncertainty = [False, True][:1]
    # Load altitude to visualizer
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        altitude_to_visualizer[altitude] = StudyVisualizerForNonStationaryTrends(
            study=CrocusSnowLoadTotal(altitude=altitude), multiprocessing=True, save_to_file=True)
    # Plot trends
    max_abs_tdrl = max([visualizer.max_abs_tdrl for visualizer in altitude_to_visualizer.values()])
    for visualizer in altitude_to_visualizer.values():
        visualizer.plot_trends(max_abs_tdrl)
    # Plot graph
    plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict(altitude_to_visualizer,
                                                                         massif_names,
                                                                         non_stationary_models_for_uncertainty,
                                                                         uncertainty_methods)


if __name__ == '__main__':
    # draw_snow_load_map(altitude=1800)
    main_results()
