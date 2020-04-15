import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad1Day, CrocusSnowLoad3Days, \
    CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, ticks_values_and_labels_for_percentages
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import ALL_ALTITUDES_WITHOUT_NAN
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod, fitmethod_to_str

import matplotlib.pyplot as plt

from projects.contrasting_trends_in_snow_loads.gorman_figures.figure1.result_from_stationary_fit import \
    ResultFromDoubleStationaryFit


class StudyVisualizerForReturnLevelChange(StudyVisualizer):

    def __init__(self, study_class, altitude, return_period=30, year_min=1959, year_middle=1989, year_max=2019,
                 save_to_file=False,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        self.return_period = return_period
        self.fit_method = fit_method

        # Load studies
        self._year_min = year_min
        self._year_middle = year_middle
        self._year_max = year_max
        self.study_before = study_class(altitude=altitude, year_min=self.year_min_before, year_max=self.year_max_before)
        self.study_after = study_class(altitude=altitude, year_min=self.year_min_after, year_max=self.year_max_after)

        # Study will always refer to study before
        super().__init__(self.study_before, save_to_file=save_to_file, show=not save_to_file)

        # Load the main part:
        self.result_from_double_stationary_fit = ResultFromDoubleStationaryFit(self.study_before,
                                                                               self.study_after,
                                                                               self.fit_method,
                                                                               self.return_period)

    @property
    def year_min_before(self):
        return self._year_min

    @property
    def year_max_before(self):
        return self._year_middle

    @property
    def year_min_after(self):
        return self._year_middle + 1

    @property
    def year_max_after(self):
        return self._year_max

    def all_plots(self):
        self.plot_return_level_change()
        self.plot_shape_values()
        self.plot_difference_returnn_level_maxima()
        self.plot_returnn_level()

    def plot_returnn_level(self):
        for result in [self.result_from_double_stationary_fit.result_before,
                       self.result_from_double_stationary_fit.result_after]:
            plot_name = 'return level {}-{}'.format(result.study.year_min,
                                                    result.study.year_max)
            label = plot_name
            self.plot_abstract(
                massif_name_to_value=result.massif_name_to_return_level,
                label=label, plot_name=plot_name,
                cmap=plt.cm.seismic)

    def plot_difference_returnn_level_maxima(self):
        for result in [self.result_from_double_stationary_fit.result_before,
                       self.result_from_double_stationary_fit.result_after]:
            plot_name = 'difference return level and maxima for {}-{}'.format(result.study.year_min,
                                                                              result.study.year_max)
            label = plot_name
            self.plot_abstract(
                massif_name_to_value=result.massif_name_to_difference_return_level_and_maxima,
                label=label, plot_name=plot_name,
                cmap=plt.cm.coolwarm)

    def plot_shape_values(self):
        for result in [self.result_from_double_stationary_fit.result_before,
                       self.result_from_double_stationary_fit.result_after]:
            plot_name = 'shape for {}-{}'.format(result.study.year_min, result.study.year_max)
            label = plot_name
            self.plot_abstract(
                massif_name_to_value=result.massif_name_to_shape,
                label=label, plot_name=plot_name, graduation=0.1,
                cmap=matplotlib.cm.get_cmap('BrBG_r'))

    def plot_return_level_change(self):
        unit = [self.study.variable_unit, '%']
        beginning = ['', 'relative']
        massif_name_to_values = [
            self.result_from_double_stationary_fit.massif_name_to_difference_return_level,
            self.result_from_double_stationary_fit.massif_name_to_relative_difference_return_level
        ]
        for u, b, massif_name_to_value in zip(unit, beginning, massif_name_to_values):
            label = 'Change {} in {} return level between \n' \
                    'a GEV fitted on {}-{} and \na GEV fitted on {}-{} ({})'.format(b,
                                                                                    self.return_period,
                                                                                    self.year_min_before,
                                                                                    self.year_max_before,
                                                                                    self.year_min_after,
                                                                                    self.year_max_after,
                                                                                    u)
            plot_name = 'Change {} in return levels'.format(b)
            self.plot_abstract(
                massif_name_to_value=massif_name_to_value,
                label=label, plot_name=plot_name)

    def plot_abstract(self, massif_name_to_value, label, plot_name, graduation=10.0, cmap=plt.cm.bwr):
        plot_name1 = '{}/{}'.format(self.study.altitude, plot_name)
        plot_name2 = '{}/{}'.format(plot_name.split()[0], plot_name)
        for plot_name in [plot_name1, plot_name2]:
            self.load_plot(cmap, graduation, label, massif_name_to_value)
            self.plot_name = plot_name
            self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True,
                                      dpi=500)
            plt.close()

    def load_plot(self, cmap, graduation, label, massif_name_to_value):
        max_abs_change = max([abs(e) for e in massif_name_to_value.values()])
        ticks, labels = ticks_values_and_labels_for_percentages(graduation=graduation, max_abs_change=max_abs_change)
        min_ratio = -max_abs_change
        max_ratio = max_abs_change
        cmap = get_shifted_map(min_ratio, max_ratio, cmap)
        massif_name_to_color = {m: get_colors([v], cmap, min_ratio, max_ratio)[0]
                                for m, v in massif_name_to_value.items()}
        ticks_values_and_labels = ticks, labels
        ax = plt.gca()
        AbstractStudy.visualize_study(ax=ax,
                                      massif_name_to_value=massif_name_to_value,
                                      massif_name_to_color=massif_name_to_color,
                                      replace_blue_by_white=True,
                                      axis_off=False,
                                      cmap=cmap,
                                      show_label=False,
                                      add_colorbar=True,
                                      show=False,
                                      vmin=min_ratio,
                                      vmax=max_ratio,
                                      ticks_values_and_labels=ticks_values_and_labels,
                                      label=label,
                                      fontsize_label=10,
                                      )
        ax.get_xaxis().set_visible(True)
        ax.set_xticks([])
        ax.set_xlabel('Altitude = {}m'.format(self.study.altitude), fontsize=15)
        ax.set_title('Fit method is {}'.format(fitmethod_to_str(self.fit_method)))
