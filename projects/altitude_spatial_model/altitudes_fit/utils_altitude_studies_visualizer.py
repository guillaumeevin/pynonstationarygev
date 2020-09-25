from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels


def load_visualizer_list(season, study_class, altitudes_list, massif_names):
    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    visualizer_list = []
    # Load all studies
    for altitudes in altitudes_list:
        # if issubclass(study_class, SimulationStudy):
        #     for ensemble_idx in list(range(14))[:1]:
        #         studies = AltitudesStudies(study_class, altitudes, season=season,
        #                                    ensemble_idx=ensemble_idx)
        #         plot_studies(massif_names, season, studies, study_class)
        # else:
        studies = AltitudesStudies(study_class, altitudes, season=season)
        visualizer = AltitudesStudiesVisualizerForNonStationaryModels(studies=studies,
                                                                      model_classes=model_classes,
                                                                      massif_names=massif_names,
                                                                      show=False,
                                                                      temporal_covariate_for_fit=None,
                                                                      # temporal_covariate_for_fit=MeanAlpsTemperatureCovariate,
                                                                      )
        visualizer_list.append(visualizer)
    compute_and_assign_max_abs(visualizer_list)

    return visualizer_list


def compute_and_assign_max_abs(visualizer_list):
    # Compute the max abs for all metrics
    d = {}
    for method_name in AltitudesStudiesVisualizerForNonStationaryModels.moment_names:
        for order in AltitudesStudiesVisualizerForNonStationaryModels.orders:
            c = (method_name, order)
            max_abs = max([
                max([abs(e) for e in v.method_name_and_order_to_d(method_name, order).values()
                     ]) for v in visualizer_list])
            d[c] = max_abs
    # Assign the max abs dictionary
    for v in visualizer_list:
        v._method_name_and_order_to_max_abs = d
    # Compute the max abs for the shape parameter
    max_abs_for_shape = max([max([abs(e) for e in v.massif_name_to_shape.values()]) for v in visualizer_list])
    for v in visualizer_list:
        v._max_abs_for_shape = max_abs_for_shape
