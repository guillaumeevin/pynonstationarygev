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
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class VisualizerForProjectionEnsemble(StudyVisualizer):

    def __init__(self, gcm_rcm_couple_to_altitude_studies: Dict[str, AltitudesStudies],
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False
                 ):
        studies = list(gcm_rcm_couple_to_altitude_studies.values())[0]
        study = studies.study
        super().__init__(study, show=show, save_to_file=not show)
        # Load one fold fit
        self.massif_name_to_massif_altitudes = {}
        self.ensemble_class_to_ensemble_fit = {}
        for ensemble_fit_class in ensemble_fit_classes:
            ensemble_fit = ensemble_fit_class(massif_names, gcm_rcm_couple_to_altitude_studies, model_classes,
                                              fit_method, temporal_covariate_for_fit,
                                              display_only_model_that_pass_gof_test,
                                              confidence_interval_based_on_delta_method)
            self.ensemble_class_to_ensemble_fit[ensemble_fit_class] = ensemble_fit

    def plot(self):
        self.plot_independent()
        self.plot_dependent()

    def plot_dependent(self):
        pass

    def plot_independent(self):
        for ensemble_fit in self.ensemble_class_to_ensemble_fit.values():
            ensemble_fit.plot()
