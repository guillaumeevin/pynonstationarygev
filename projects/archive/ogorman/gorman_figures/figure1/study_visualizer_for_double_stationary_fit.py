import matplotlib

from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod

import matplotlib.pyplot as plt

from projects.archive.ogorman.gorman_figures.figure1.result_from_stationary_fit import ResultFromDoubleStationaryFit


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
                fit_method=self.fit_method,
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
                label=label, plot_name=plot_name,
                fit_method=self.fit_method)
