from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, rcp_scenarios
from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.projected_snowfall.elevation_temporal_model_for_projections.visualizer_for_projection_ensemble import \
    VisualizerForProjectionEnsemble


def load_projected_visualizer_list(gcm_rcm_couples, ensemble_fit_classes,
                                   season, study_class, altitudes_list, massif_names, model_must_pass_the_test=False,
                                   scenario=AdamontScenario.rcp85,
                                   temporal_covariate_for_fit=None, **kwargs_study):
    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    assert scenario in rcp_scenarios
    visualizer_list = []
    # Load all studies
    for altitudes in altitudes_list:
        gcm_rcm_couple_to_altitude_studies = {}
        for gcm_rcm_couple in gcm_rcm_couples:
            studies = AltitudesStudies(study_class, altitudes, season=season,
                                       scenario=scenario, gcm_rcm_couple=gcm_rcm_couple, **kwargs_study)
            gcm_rcm_couple_to_altitude_studies[gcm_rcm_couple] = studies
        visualizer = VisualizerForProjectionEnsemble(gcm_rcm_couple_to_altitude_studies=gcm_rcm_couple_to_altitude_studies,
                                                     model_classes=model_classes,
                                                     ensemble_fit_classes=ensemble_fit_classes,
                                                     massif_names=massif_names,
                                                     show=False,
                                                     temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                     confidence_interval_based_on_delta_method=False,
                                                     display_only_model_that_pass_gof_test=model_must_pass_the_test
                                                     )
        visualizer_list.append(visualizer)

    return visualizer_list


