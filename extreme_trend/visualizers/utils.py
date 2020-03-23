from collections import OrderedDict

from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


def load_altitude_to_visualizer(altitudes, massif_names, model_subsets_for_uncertainty, study_class,
                                uncertainty_methods,
                                study_visualizer_class=StudyVisualizerForNonStationaryTrends,
                                save_to_file=True,
                                multiprocessing=True):
    fit_method = TemporalMarginFitMethod.extremes_fevd_mle
    altitude_to_visualizer = OrderedDict()
    for altitude in altitudes:
        study = study_class(altitude=altitude, multiprocessing=multiprocessing)
        study_visualizer = study_visualizer_class(study=study, multiprocessing=multiprocessing,
                                                  save_to_file=save_to_file, uncertainty_massif_names=massif_names,
                                                  uncertainty_methods=uncertainty_methods,
                                                  model_subsets_for_uncertainty=model_subsets_for_uncertainty,
                                                  fit_method=fit_method, select_only_acceptable_shape_parameter=True,
                                                  fit_gev_only_on_non_null_maxima=False,
                                                  fit_only_time_series_with_ninety_percent_of_non_null_values=True)
        altitude_to_visualizer[altitude] = study_visualizer

    return altitude_to_visualizer
