from typing import List

from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import \
    get_altitude_class_from_altitudes
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plots.plot_histogram_altitude_studies import \
    plot_histogram_all_trends_against_altitudes, plot_shoe_plot_changes_against_altitude
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.utils_altitude_studies_visualizer import compute_and_assign_max_abs
from projects.projected_snowfall.elevation_temporal_model_for_projections.independent_ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit


class VisualizerForProjectionEnsemble(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 gcm_to_year_min_and_year_max=None,
                 ):
        self.gcm_rcm_couples = gcm_rcm_couples
        self.massif_names = massif_names
        self.ensemble_fit_classes = ensemble_fit_classes

        # Load all studies
        altitude_class_to_gcm_couple_to_studies = {}
        for altitudes in altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
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
            if len(gcm_rcm_couple_to_studies) == 0:
                print('No valid studies for the following couples:', self.gcm_rcm_couples)
            altitude_class_to_gcm_couple_to_studies[altitude_class] = gcm_rcm_couple_to_studies

        # Load ensemble fit
        self.altitude_class_to_ensemble_class_to_ensemble_fit = {}
        for altitude_class, gcm_rcm_couple_to_studies in altitude_class_to_gcm_couple_to_studies.items():
            ensemble_class_to_ensemble_fit = {}
            for ensemble_fit_class in ensemble_fit_classes:
                ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_studies, model_classes,
                                                  fit_method, temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method,
                                                  remove_physically_implausible_models)
                ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit
            self.altitude_class_to_ensemble_class_to_ensemble_fit[altitude_class] = ensemble_class_to_ensemble_fit

    def plot(self):
        if IndependentEnsembleFit in self.ensemble_fit_classes:
            # Set max abs
            visualizer_list = []
            for ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
                visualizer_list.extend(list(ensemble_fit.gcm_rcm_couple_to_visualizer.values()))
            # Potentially I could add more visualizer here...
            method_name_and_order_to_max_abs, max_abs_for_shape = compute_and_assign_max_abs(visualizer_list)
            # Assign the same max abs for the 
            for ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
                for v in ensemble_fit.merge_function_name_to_visualizer.values():
                    v._max_abs_for_shape = max_abs_for_shape
                    v._method_name_and_order_to_max_abs = method_name_and_order_to_max_abs
            # Plot
            self.plot_independent()

    def plot_independent(self):
        with_significance = False
        # Aggregated at gcm_rcm_level plots
        merge_keys = [IndependentEnsembleFit.Median_merge, IndependentEnsembleFit.Mean_merge]
        keys = self.gcm_rcm_couples + merge_keys
        # Only plot Mean for speed
        keys = [IndependentEnsembleFit.Mean_merge]
        for key in keys:
            visualizer_list = [independent_ensemble_fit.gcm_rcm_couple_to_visualizer[key]
                               if key in self.gcm_rcm_couples
                               else independent_ensemble_fit.merge_function_name_to_visualizer[key]
                               for independent_ensemble_fit in self.ensemble_fits(IndependentEnsembleFit)
                               ]
            if key in merge_keys:
                for v in visualizer_list:
                    v.studies.study.gcm_rcm_couple = (key, "merge")
            for v in visualizer_list:
                v.plot_moments()
            plot_histogram_all_trends_against_altitudes(self.massif_names, visualizer_list, with_significance=with_significance)
            for relative in [True, False]:
                plot_shoe_plot_changes_against_altitude(self.massif_names, visualizer_list, relative=relative, with_significance=with_significance)

    def ensemble_fits(self, ensemble_class):
        return [ensemble_class_to_ensemble_fit[ensemble_class]
                for ensemble_class_to_ensemble_fit
                in self.altitude_class_to_ensemble_class_to_ensemble_fit.values()]

    def plot_together(self):
        pass
