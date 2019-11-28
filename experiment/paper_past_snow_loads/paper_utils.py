from collections import OrderedDict

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends

paper_altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode, CrocusSnowLoad3Days]


def load_altitude_to_visualizer(altitudes, massif_names, non_stationary_uncertainty, study_class, uncertainty_methods):
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        altitude_to_visualizer[altitude] = StudyVisualizerForNonStationaryTrends(
            study=study_class(altitude=altitude), multiprocessing=True, save_to_file=True,
            uncertainty_massif_names=massif_names, uncertainty_methods=uncertainty_methods,
            non_stationary_contexts=non_stationary_uncertainty)
    return altitude_to_visualizer
