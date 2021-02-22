from collections import Counter
from math import ceil, floor
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, ticks_values_and_labels_for_percentages, get_half_colormap, ticks_values_and_labels_for_positive_value, \
    get_inverse_colormap, get_cmap_with_inverted_blue_and_green_channels, remove_the_extreme_colors
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION, ALL_ALTITUDES_WITHOUT_NAN
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import \
    get_altitude_group_from_altitudes, HighAltitudeGroup, VeyHighAltitudeGroup, MidAltitudeGroup
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit
from projects.altitude_spatial_model.altitudes_fit.plots.plot_histogram_altitude_studies import \
    plot_histogram_all_trends_against_altitudes, plot_shoe_plot_changes_against_altitude
from projects.altitude_spatial_model.altitudes_fit.utils_altitude_studies_visualizer import compute_and_assign_max_abs
from projects.projected_snowfall.elevation_temporal_model_for_projections.ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class MetaVisualizerForProjectionEnsemble(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 ):
        self.gcm_rcm_couples = gcm_rcm_couples
        self.massif_names = massif_names
        self.ensemble_fit_classes = ensemble_fit_classes

        # Load all studies
        altitude_group_to_gcm_couple_to_studies = {}
        for altitudes in altitudes_list:
            altitude_group = get_altitude_group_from_altitudes(altitudes)
            gcm_rcm_couple_to_studies = {}
            for gcm_rcm_couple in gcm_rcm_couples:
                studies = AltitudesStudies(study_class, altitudes, season=season,
                                           scenario=scenario, gcm_rcm_couple=gcm_rcm_couple)
                gcm_rcm_couple_to_studies[gcm_rcm_couple] = studies
            altitude_group_to_gcm_couple_to_studies[altitude_group] = gcm_rcm_couple_to_studies

        # Load ensemble fit
        self.altitude_group_to_ensemble_class_to_ensemble_fit = {}
        for altitude_group, gcm_rcm_couple_to_studies in altitude_group_to_gcm_couple_to_studies.items():
            ensemble_class_to_ensemble_fit = {}
            for ensemble_fit_class in ensemble_fit_classes:
                ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_studies, model_classes,
                                                  fit_method, temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method,
                                                  remove_physically_implausible_models)
                ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit
            self.altitude_group_to_ensemble_class_to_ensemble_fit[altitude_group] = ensemble_class_to_ensemble_fit

    def plot(self):
        if IndependentEnsembleFit in self.ensemble_fit_classes:
            # Set max abs
            visualizer_list = []
            for ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
                visualizer_list.extend(list(ensemble_fit.gcm_rcm_couple_to_visualizer.values()))
            # Potentially I could add more visualizer here...
            compute_and_assign_max_abs(visualizer_list)
            # Plot
            self.plot_independent()

    def plot_independent(self):
        with_significance = False
        # Individual plots
        for independent_ensemble_fit in self.ensemble_fits(IndependentEnsembleFit):
            print(independent_ensemble_fit)
            for c, v in independent_ensemble_fit.gcm_rcm_couple_to_visualizer.items():
                print(c, v.altitude_group)
                v.plot_moments()
        # Aggregated at gcm_rcm_level plots
        for gcm_rcm_couple in self.gcm_rcm_couples:
            visualizer_list = [independent_ensemble_fit.gcm_rcm_couple_to_visualizer[gcm_rcm_couple]
                               for independent_ensemble_fit in self.ensemble_fits(IndependentEnsembleFit)
                               ]
            plot_histogram_all_trends_against_altitudes(self.massif_names, visualizer_list, with_significance=with_significance)
            for relative in [True, False]:
                plot_shoe_plot_changes_against_altitude(self.massif_names, visualizer_list, relative=relative, with_significance=with_significance)

    def ensemble_fits(self, ensemble_class):
        return [ensemble_class_to_ensemble_fit[ensemble_class]
                for ensemble_class_to_ensemble_fit
                in self.altitude_group_to_ensemble_class_to_ensemble_fit.values()]

    def plot_together(self):
        pass
