from collections import OrderedDict
from typing import List

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.one_fold_fit.altitude_group import get_altitude_group_from_altitudes


class VisualizerForProjectionEnsemble(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 gcm_to_year_min_and_year_max=None,
                 interval_str_prefix='',
                 safran_study_class=None,
                 param_name_to_climate_coordinates_with_effects=None,
                 linear_effects=(False, False, False),
                 ):
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.study_class = study_class
        self.safran_study_class = safran_study_class
        self.interval_str_prefix = interval_str_prefix
        self.altitudes_list = altitudes_list
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.scenario = scenario
        self.gcm_rcm_couples = gcm_rcm_couples
        self.massif_names = massif_names
        self.ensemble_fit_classes = ensemble_fit_classes

        # Some checks
        if gcm_to_year_min_and_year_max is not None:
            for gcm, years in gcm_to_year_min_and_year_max.items():
                assert isinstance(gcm, str), gcm
                assert len(years) == 2, years

        # Load all studies
        altitude_group_to_gcm_couple_to_studies = OrderedDict()
        for altitudes in altitudes_list:
            altitude_group = get_altitude_group_from_altitudes(altitudes)
            gcm_rcm_couple_to_studies = self.load_gcm_rcm_couple_to_studies(altitudes, gcm_rcm_couples,
                                                                            gcm_to_year_min_and_year_max,
                                                                            safran_study_class, scenario, season,
                                                                            study_class)
            altitude_group_to_gcm_couple_to_studies[altitude_group] = gcm_rcm_couple_to_studies

        # Load ensemble fit
        self.altitude_group_to_ensemble_class_to_ensemble_fit = OrderedDict()
        for altitude_group, gcm_rcm_couple_to_studies in altitude_group_to_gcm_couple_to_studies.items():
            ensemble_class_to_ensemble_fit = {}
            for ensemble_fit_class in ensemble_fit_classes:
                ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_studies, model_classes,
                                                  fit_method, temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method,
                                                  remove_physically_implausible_models,
                                                  param_name_to_climate_coordinates_with_effects,
                                                  linear_effects)
                ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit
            self.altitude_group_to_ensemble_class_to_ensemble_fit[altitude_group] = ensemble_class_to_ensemble_fit

    @classmethod
    def load_gcm_rcm_couple_to_studies(cls, altitudes, gcm_rcm_couples, gcm_to_year_min_and_year_max,
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

    def ensemble_fits(self, ensemble_class):
        """Return the ordered ensemble fit for a given ensemble class (in the order of the altitudes)"""
        return [ensemble_class_to_ensemble_fit[ensemble_class]
                for ensemble_class_to_ensemble_fit
                in self.altitude_group_to_ensemble_class_to_ensemble_fit.values()]

