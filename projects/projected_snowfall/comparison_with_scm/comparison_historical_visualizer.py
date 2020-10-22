import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_color_from_gcm_rcm_couple
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


class ComparisonHistoricalVisualizer(StudyVisualizer):

    def __init__(self, scm_study: AbstractStudy,
                 adamont_studies: AdamontStudies,
                 show=False,
                 massif_names=None,
                 ):
        super().__init__(scm_study, show=show, save_to_file=not show)
        self.scm_study = scm_study
        self.adamont_studies = adamont_studies
        if massif_names is None:
            massif_names = scm_study.study_massif_names
        self.massif_names = massif_names

    def get_values(self, study_method, massif_name):
        """
        Return an array of size (nb_ensembles + 1) x nb_observations
        :param study_method:
        :param massif_name:
        :return:
        """
        values = [self.scm_study.__getattribute__(study_method)[massif_name]]
        for study in self.adamont_studies.study_list:
            values.append(study.__getattribute__(study_method)[massif_name])
        return np.array(values)

    def plot_comparison(self, plot_maxima=True):
        if plot_maxima:
            study_method = 'massif_name_to_annual_maxima'
        else:
            study_method = 'massif_name_to_daily_time_series'
        value_name = study_method.split('to_')[1]
        print('Nb massifs', len(self.massif_names))
        for massif_name in self.massif_names:
            values = self.get_values(study_method, massif_name)
            plot_name = value_name + ' for {}'.format(massif_name.replace('_', '-'))
            self.shoe_plot_comparison(values, plot_name)

    def shoe_plot_comparison(self, values, plot_name):
        ax = plt.gca()
        width = 10
        positions = [i * width * 2 for i in range(len(values))]
        labels = ['Reanalysis'] + [' / '.join(couple) for couple in self.adamont_studies.gcm_rcm_couples]
        colors = ['black'] + [get_color_from_gcm_rcm_couple(couple) for couple in self.adamont_studies.gcm_rcm_couples]
        # Permute values, labels & colors, based on the mean values
        print(values.shape)
        mean_values = np.mean(values, axis=1)
        print(mean_values.shape)
        index_to_sort = np.argsort(mean_values)
        colors = [colors[i] for i in index_to_sort]
        labels = [labels[i] for i in index_to_sort]
        values = [values[i] for i in index_to_sort]
        # Add boxplot with legend
        bplot = ax.boxplot(values, positions=positions, widths=width, patch_artist=True, showmeans=True)
        for color, patch in zip(colors, bplot['boxes']):
            patch.set_facecolor(color)
        custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
        ax.legend(custom_lines, labels, ncol=2)
        ax.set_xticks([])
        ax.set_xlim([min(positions) - width, max(positions) + width])
        ylabel = 'Annual maxima' if 'maxima' in plot_name else 'daily values'
        ax.set_ylabel('{} of {} ({})'.format(ylabel,
                                             SCM_STUDY_CLASS_TO_ABBREVIATION[type(self.study)],
                                                        self.study.variable_unit))

        self.plot_name = '{}'.format(plot_name)
        self.show_or_save_to_file(add_classic_title=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()


