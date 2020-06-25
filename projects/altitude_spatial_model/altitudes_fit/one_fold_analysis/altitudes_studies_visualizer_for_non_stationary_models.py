from typing import List
import matplotlib.pyplot as plt

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None):
        study = studies.study
        self.massif_names = massif_names if massif_names is not None else self.study.study_massif_names
        self.studies = studies
        self.non_stationary_models = model_classes
        super().__init__(study, show=show, save_to_file=not show)
        self.massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            dataset = studies.spatio_temporal_dataset(massif_name=massif_name)
            old_fold_fit = OneFoldFit(dataset, model_classes)
            self.massif_name_to_one_fold_fit[massif_name] = old_fold_fit

    def plot_mean(self):
        self.plot_general('mean')

    def plot_relative_change(self):
        self.plot_general('relative_changes_in_the_mean')

    def plot_general(self, method_name):
        ax = plt.gca()
        min_altitude, *_, max_altitude = self.studies.altitudes
        altitudes = np.linspace(min_altitude, max_altitude, num=50)
        for massif_id, massif_name in enumerate(self.massif_names):
            old_fold_fit = self.massif_name_to_one_fold_fit[massif_name]
            values = old_fold_fit.__getattribute__(method_name)(altitudes)
            plot_against_altitude(altitudes, ax, massif_id, massif_name, values)
        # Plot settings
        ax.legend(prop={'size': 7}, ncol=3)
        moment = ' '.join(method_name.split('_'))
        plot_name = '{} annual maxima of {}'.format(moment, SCM_STUDY_CLASS_TO_ABBREVIATION[self.studies.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        # lim_down, lim_up = ax.get_ylim()
        # lim_up += (lim_up - lim_down) / 3
        # ax.set_ylim([lim_down, lim_up])
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show)
        ax.clear()
