from collections import OrderedDict

from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod


def load_altitude_to_visualizer(altitudes, massif_names, non_stationary_uncertainty, study_class, uncertainty_methods,
                                study_visualizer_class=StudyVisualizerForNonStationaryTrends,
                                save_to_file=True):
    fit_method = TemporalMarginFitMethod.extremes_fevd_gmle
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        altitude_to_visualizer[altitude] = study_visualizer_class(
            study=study_class(altitude=altitude), multiprocessing=True, save_to_file=save_to_file,
            uncertainty_massif_names=massif_names, uncertainty_methods=uncertainty_methods,
            non_stationary_contexts=non_stationary_uncertainty, fit_method=fit_method)
    return altitude_to_visualizer
