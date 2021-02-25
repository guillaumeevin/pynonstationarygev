from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def load_visualizer_list(season, study_class, altitudes_list, massif_names, model_must_pass_the_test=True, **kwargs_study):
    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    visualizer_list = []
    # Load all studies
    for altitudes in altitudes_list:
        studies = AltitudesStudies(study_class, altitudes, season=season, **kwargs_study)
        visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies=studies,
                                                                      model_classes=model_classes,
                                                                      massif_names=massif_names,
                                                                      show=False,
                                                                      temporal_covariate_for_fit=None,
                                                                      confidence_interval_based_on_delta_method=False,
                                                                      display_only_model_that_pass_anderson_test=model_must_pass_the_test
                                                                      )
        visualizer_list.append(visualizer)
    compute_and_assign_max_abs(visualizer_list)

    return visualizer_list


def compute_and_assign_max_abs(visualizer_list):
    # Compute the max abs for all metrics
    method_name_and_order_to_max_abs = {}
    for method_name in AltitudesStudiesVisualizerForNonStationaryModels.moment_names:
        for order in AltitudesStudiesVisualizerForNonStationaryModels.orders:
            c = (method_name, order)
            max_abs = max([
                max([abs(e) for e in v.method_name_and_order_to_d(method_name, order).values()
                     ]) for v in visualizer_list])
            if max_abs == 0:
                epsilon = 0.1
                max_abs = epsilon
            method_name_and_order_to_max_abs[c] = max_abs
    # Assign the max abs dictionary
    for v in visualizer_list:
        v._method_name_and_order_to_max_abs = method_name_and_order_to_max_abs
    # Compute the max abs for the shape parameter
    max_abs_for_shape = max([max([abs(e) for e in v.massif_name_to_shape.values()]) for v in visualizer_list])
    for v in visualizer_list:
        v._max_abs_for_shape = max_abs_for_shape
    return method_name_and_order_to_max_abs, max_abs_for_shape
