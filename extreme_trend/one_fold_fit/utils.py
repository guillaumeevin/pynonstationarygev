from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.ensemble_fit.visualizer_non_stationary_ensemble import VisualizerNonStationaryEnsemble


def load_sub_visualizer(altitudes, display_only_model_that_pass_gof_test, fit_method, gcm_rcm_couples, linear_effects,
                        massif_name_to_model_class, massif_name_to_param_name_to_climate_coordinates_with_effects,
                        massif_names, remove_physically_implausible_models, safran_study_class, scenario, season,
                        study_class, temporal_covariate_for_fit):
    gcm_rcm_couple_to_studies = load_gcm_rcm_couple_to_studies(altitudes,
                                                               gcm_rcm_couples,
                                                               None, safran_study_class,
                                                               scenario, season,
                                                               study_class)
    sub_visualizer = VisualizerNonStationaryEnsemble(gcm_rcm_couple_to_studies,
                                                     massif_name_to_model_class,
                                                     False,
                                                     massif_names, fit_method,
                                                     temporal_covariate_for_fit,
                                                     display_only_model_that_pass_gof_test,
                                                     False,
                                                     remove_physically_implausible_models,
                                                     massif_name_to_param_name_to_climate_coordinates_with_effects,
                                                     linear_effects
                                                     )
    return sub_visualizer

def load_gcm_rcm_couple_to_studies(altitudes, gcm_rcm_couples, gcm_to_year_min_and_year_max,
                                   safran_study_class, scenario, season, study_class,
                                   year_max_for_safran_study=None):
    gcm_rcm_couple_to_studies = {}
    for gcm_rcm_couple in gcm_rcm_couples:
        if gcm_to_year_min_and_year_max is None:
            kwargs_study = {}
        else:
            gcm = gcm_rcm_couple[0]
            if gcm not in gcm_to_year_min_and_year_max:
                # It means that for this gcm and scenario,
                # there is not enough data (less than 30 years) for the fit
                continue
            year_min, year_max = gcm_to_year_min_and_year_max[gcm]
            kwargs_study = {'year_min': year_min, 'year_max': year_max}
        studies = AltitudesStudies(study_class, altitudes, season=season,
                                   scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                   **kwargs_study)
        gcm_rcm_couple_to_studies[gcm_rcm_couple] = studies
    # Potentially add the observations
    if safran_study_class is not None:
        if year_max_for_safran_study is not None:
            studies = AltitudesStudies(safran_study_class, altitudes, season=season,
                                       year_max=year_max_for_safran_study)
        else:
            studies = AltitudesStudies(safran_study_class, altitudes, season=season)
        gcm_rcm_couple_to_studies[(None, None)] = studies
    if len(gcm_rcm_couple_to_studies) == 0:
        print('No valid studies for the following couples:', gcm_rcm_couples)
    return gcm_rcm_couple_to_studies
