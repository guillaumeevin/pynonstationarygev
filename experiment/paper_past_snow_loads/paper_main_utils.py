from collections import OrderedDict
from enum import Enum

from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod


def load_altitude_to_visualizer(altitudes, massif_names, model_subsets_for_uncertainty, study_class,
                                uncertainty_methods,
                                study_visualizer_class=StudyVisualizerForNonStationaryTrends,
                                save_to_file=True):
    fit_method = TemporalMarginFitMethod.extremes_fevd_mle
    select_only_acceptable_shape_parameter = True
    print('Fit method: {}, Select only acceptable shape parameter: {}'
          .format(fit_method, select_only_acceptable_shape_parameter))
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        altitude_to_visualizer[altitude] = study_visualizer_class(
            study=study_class(altitude=altitude), multiprocessing=True, save_to_file=save_to_file,
            uncertainty_massif_names=massif_names, uncertainty_methods=uncertainty_methods,
            model_subsets_for_uncertainty=model_subsets_for_uncertainty, fit_method=fit_method,
            select_only_acceptable_shape_parameter=select_only_acceptable_shape_parameter)
    return altitude_to_visualizer


