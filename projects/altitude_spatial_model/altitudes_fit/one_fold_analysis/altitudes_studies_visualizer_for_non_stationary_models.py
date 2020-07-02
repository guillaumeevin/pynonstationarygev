from typing import List
import matplotlib.pyplot as plt

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(studies.study, show=show, save_to_file=not show)
        self.massif_names = massif_names if massif_names is not None else self.study.study_massif_names
        self.studies = studies
        self.non_stationary_models = model_classes
        self.fit_method = fit_method
        self.massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            dataset = studies.spatio_temporal_dataset(massif_name=massif_name)
            old_fold_fit = OneFoldFit(massif_name, dataset, model_classes, self.fit_method)
            self.massif_name_to_one_fold_fit[massif_name] = old_fold_fit

    def plot_moments(self):
        for method_name in ['moment', 'changes_in_the_moment', 'relative_changes_in_the_moment']:
            for order in [1, 2, None]:
                self.plot_general(method_name, order)

    def plot_general(self, method_name, order):
        ax = plt.gca()
        min_altitude, *_, max_altitude = self.studies.altitudes
        altitudes_plot = np.linspace(min_altitude, max_altitude, num=50)
        for massif_id, massif_name in enumerate(self.massif_names):
            massif_altitudes = self.studies.massif_name_to_altitudes[massif_name]
            ind = (min(massif_altitudes) <= altitudes_plot) & (altitudes_plot <= max(massif_altitudes))
            massif_altitudes_plot = altitudes_plot[ind]
            one_fold_fit = self.massif_name_to_one_fold_fit[massif_name]
            values = one_fold_fit.__getattribute__(method_name)(massif_altitudes_plot, order=order)
            plot_against_altitude(massif_altitudes_plot, ax, massif_id, massif_name, values)
        # Plot settings
        ax.legend(prop={'size': 7}, ncol=3)
        moment = ' '.join(method_name.split('_'))
        moment = moment.replace('moment', '{} in 2019'.format(OneFoldFit.get_moment_str(order=order)))
        plot_name = '{}/Model {} annual maxima of {}'.format(OneFoldFit.folder_for_plots, moment,
                                                             SCM_STUDY_CLASS_TO_ABBREVIATION[self.studies.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        # lim_down, lim_up = ax.get_ylim()
        # lim_up += (lim_up - lim_down) / 3
        # ax.set_ylim([lim_down, lim_up])
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show)
        ax.clear()

    def plot_abstract_fast(self, massif_name_to_value, label, graduation=10.0, cmap=plt.cm.coolwarm, add_x_label=True,
                           negative_and_positive_values=True, massif_name_to_text=None):
        plot_name = '{}/{}'.format(OneFoldFit.folder_for_plots, label)
        self.plot_map(cmap, self.fit_method, graduation, label, massif_name_to_value, plot_name, add_x_label,
                      negative_and_positive_values,
                      massif_name_to_text)

    @property
    def massif_name_to_shape(self):
        return {massif_name: one_fold_fit.best_shape
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    @property
    def massif_name_to_name(self):
        return {massif_name: one_fold_fit.best_name
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    def plot_best_coef_maps(self):
        for param_name in GevParams.PARAM_NAMES:
            coordinate_names = [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_T]
            dim_to_coordinate_name = dict(zip([0, 1], coordinate_names))
            for dim in [0, 1, (0, 1)]:
                coordinate_name = LinearCoef.coefficient_name(dim, dim_to_coordinate_name)
                for degree in range(3):
                    coef_name = ' '.join([param_name + coordinate_name + str(degree)])
                    massif_name_to_best_coef = {}
                    for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
                        best_coef = one_fold_fit.best_coef(param_name, dim, degree)
                        if best_coef is not None:
                            massif_name_to_best_coef[massif_name] = best_coef

                    if len(massif_name_to_best_coef) > 0:
                        for evaluate_coordinate in [False, True]:
                            if evaluate_coordinate:
                                coef_name += 'evaluated at coordinates'
                                for m in massif_name_to_best_coef.values():
                                    if AbstractCoordinates.COORDINATE_X in coordinate_name:
                                        massif_name_to_best_coef[m] *= np.power(1000, degree)
                                    if AbstractCoordinates.COORDINATE_T in coordinate_name:
                                        massif_name_to_best_coef[m] *= np.power(1000, degree)
                            self.plot_best_coef_map(coef_name.replace('_', ''), massif_name_to_best_coef)

    def plot_best_coef_map(self, coef_name, massif_name_to_best_coef):
        values = list(massif_name_to_best_coef.values())
        graduation = (max(values) - min(values)) / 6
        print(coef_name, graduation, max(values), min(values))
        negative_and_positive_values = (max(values) > 0) and (min(values) < 0)
        self.plot_abstract_fast(massif_name_to_best_coef,
                                label='{}/Coef/{} plot for {} {}'.format(OneFoldFit.folder_for_plots,
                                                                         coef_name,
                                                                 SCM_STUDY_CLASS_TO_ABBREVIATION[type(self.study)],
                                                                 self.study.variable_unit),
                                add_x_label=False, graduation=graduation, massif_name_to_text=self.massif_name_to_name,
                                negative_and_positive_values=negative_and_positive_values)

    def plot_shape_map(self):
        self.plot_abstract_fast(self.massif_name_to_shape,
                                label='Shape plot for {} {}'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[type(self.study)],
                                                                    self.study.variable_unit),
                                add_x_label=False, graduation=0.1, massif_name_to_text=self.massif_name_to_name)
