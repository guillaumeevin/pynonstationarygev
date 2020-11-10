import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_color_from_gcm_rcm_couple, \
    gcm_rcm_couple_to_str, gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_studies import AdamontStudies
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION, ADAMONT_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies


class ComparisonHistoricalVisualizer(StudyVisualizer):

    def __init__(self, scm_study: AbstractStudy,
                 adamont_studies: AdamontStudies,
                 show=False,
                 massif_names=None,
                 ):
        super().__init__(adamont_studies.study, show=show, save_to_file=not show)
        self.scm_study = scm_study
        self.altitude = self.scm_study.altitude
        self.adamont_studies = adamont_studies
        if massif_names is None:
            massif_names = scm_study.study_massif_names
        self.massif_names = massif_names

    def get_values(self, study_method, massif_name):
        """
        Return an array "values" of size (nb_ensembles + 1) x nb_observations
        Return gcm_rcm_couples of size nb_ensembles
        :param study_method:
        :param massif_name:
        :return:
        """
        values = [self.scm_study.__getattribute__(study_method)[massif_name]]
        gcm_rcm_couples = []
        for gcm_rcm_couple, study in self.adamont_studies.gcm_rcm_couple_to_study.items():
            try:
                values.append(study.__getattribute__(study_method)[massif_name])
                gcm_rcm_couples.append(gcm_rcm_couple)
            except KeyError:
                pass
        return np.array(values), gcm_rcm_couples

    def compute_bias_list_in_the_mean(self, massif_name, relative, study_method):
        values, gcm_rcm_couples = self.get_values(study_method, massif_name)
        mean_values = np.mean(values, axis=1)
        bias_in_the_mean = (mean_values - mean_values[0])[1:]
        if relative:
            bias_in_the_mean *= 100 / mean_values[0]
        return bias_in_the_mean, gcm_rcm_couples

    # Map massif name to bias list (ordered by the gcm_rcm_couples)

    def massif_name_to_bias_list_in_the_mean(self, plot_maxima=True, relative=False):
        study_method = self.get_study_method(plot_maxima)
        massif_name_to_bias_list = {}
        for massif_name in self.massif_names:
            bias_list, gcm_rcm_couples = self.compute_bias_list_in_the_mean(massif_name, relative, study_method)
            massif_name_to_bias_list[massif_name] = bias_list
        return massif_name_to_bias_list

    @property
    def massif_name_to_rank(self):
        massif_name_to_rank = {}
        for massif_name, bias_list in self.massif_name_to_bias_list_in_the_mean(plot_maxima=True).items():
            # Count the number of bias negative
            nb_of_negative = sum([b < 0 for b in bias_list])
            # Rank starts to 1
            massif_name_to_rank[massif_name] = float(1 + nb_of_negative)
        return massif_name_to_rank

    # Map gcm_rcm_couple to bias list (ordered by the massif_names)

    @cached_property
    def gcm_rcm_couple_to_bias_list_in_the_mean_maxima(self):
        return self.gcm_rcm_couple_to_bias_list_for_the_mean(plot_maxima=True, relative=False)

    @cached_property
    def gcm_rcm_couple_to_relative_bias_list_in_the_mean_maxima(self):
        return self.gcm_rcm_couple_to_bias_list_for_the_mean(plot_maxima=True, relative=True)

    def gcm_rcm_couple_to_bias_list_for_the_mean(self, plot_maxima=True, relative=False):
        study_method = self.get_study_method(plot_maxima)
        gcm_rcm_couple_to_bias_list = {couple: [] for couple in self.adamont_studies.gcm_rcm_couples}
        for massif_name in self.massif_names:
            bias, gcm_rcm_couples = self.compute_bias_list_in_the_mean(massif_name, relative, study_method)
            for b, couple in zip(bias, gcm_rcm_couples):
                gcm_rcm_couple_to_bias_list[couple].append(b)
        return gcm_rcm_couple_to_bias_list

    def plot_comparison(self, plot_maxima=True):
        study_method = self.get_study_method(plot_maxima)
        value_name = study_method.split('to_')[1]
        for massif_name in self.massif_names:
            values, gcm_rcm_couples = self.get_values(study_method, massif_name)
            plot_name = value_name + ' for {}'.format(massif_name.replace('_', '-'))
            self.shoe_plot_comparison(values, gcm_rcm_couples, plot_name)

    def get_study_method(self, plot_maxima):
        if plot_maxima:
            study_method = 'massif_name_to_annual_maxima'
        else:
            study_method = 'massif_name_to_daily_time_series'
        return study_method

    def shoe_plot_comparison(self, values, gcm_rcm_couples, plot_name):
        ax = plt.gca()
        width = 10
        positions = [i * width * 2 for i in range(len(values))]
        labels = ['Reanalysis'] + [gcm_rcm_couple_to_str(couple) for couple in gcm_rcm_couples]
        colors = ['black'] + [get_color_from_gcm_rcm_couple(couple) for couple in gcm_rcm_couples]
        # Permute values, labels & colors, based on the mean values
        mean_values = np.mean(values, axis=1)
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
                                             ADAMONT_STUDY_CLASS_TO_ABBREVIATION[type(self.study)],
                                             self.study.variable_unit))

        self.plot_name = 'comparison/{}/{}'.format(self.scm_study.altitude, plot_name)
        self.show_or_save_to_file(add_classic_title=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()

    def shoe_plot_bias_maxima_comparison(self):
        couples = list(self.gcm_rcm_couple_to_bias_list_in_the_mean_maxima.keys())
        values = list(self.gcm_rcm_couple_to_bias_list_in_the_mean_maxima.values())
        colors = [gcm_rcm_couple_to_color[couple] for couple in couples]
        labels = [gcm_rcm_couple_to_str(couple) for couple in couples]

        ax = plt.gca()
        width = 10
        positions = [i * width * 2 for i in range(len(values))]
        bplot = ax.boxplot(values, positions=positions, widths=width, patch_artist=True, showmeans=True)
        for color, patch in zip(colors, bplot['boxes']):
            patch.set_facecolor(color)
        custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
        ax.legend(custom_lines, labels, ncol=2)
        ax.set_xticks([])
        plot_name = 'Mean bias w.r.t to the reanalysis at {} m '.format(self.altitude)
        ax.set_ylabel(plot_name)
        ax.set_xlim([min(positions) - width, max(positions) + width])
        self.plot_name = 'altitude_comparison/{}'.format(plot_name)
        self.show_or_save_to_file(add_classic_title=False, no_title=True, tight_layout=True)
        ax.clear()
        plt.close()

    def plot_map_with_the_rank(self):
        massif_name_to_value = self.massif_name_to_rank
        max_abs_change = self.adamont_studies.nb_ensemble_members + 1
        ylabel = 'Rank of the mean maxima\n,' \
                 'which is between 1 (lowest) and {} (largest)'.format(max_abs_change)
        plot_name = ylabel
        self.plot_map(cmap=plt.cm.coolwarm, graduation=1,
                      label=ylabel,
                      massif_name_to_value=massif_name_to_value,
                      plot_name=plot_name, add_x_label=True,
                      negative_and_positive_values=False,
                      altitude=self.altitude,
                      add_colorbar=True,
                      max_abs_change=max_abs_change,
                      massif_name_to_text={m: str(v) for m, v in massif_name_to_value.items()},
                      # xlabel=self.altitude_group.xlabel,
                      )

    def plot_map_with_the_bias_in_the_mean(self, relative=True):
        pass
