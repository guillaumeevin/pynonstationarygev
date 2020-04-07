from collections import OrderedDict

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, shiftedColorMap, ticks_values_and_labels_for_percentages
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod

import matplotlib.pyplot as plt

from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import StudyVisualizerForNonStationaryTrends


class StudyVisualizerForReturnLevelChange(StudyVisualizer):

    def __init__(self, study_class, altitude, return_period=30, year_min=1959, year_middle=1989, year_max=2019, save_to_file=False,
                 fit_method=TemporalMarginFitMethod.extremes_fevd_mle):
        self.fit_method = fit_method
        self._year_min = year_min
        self._year_middle = year_middle
        self._year_max = year_max
        self.study_before = study_class(altitude=altitude, year_min=self.year_min_before, year_max=self.year_max_before)
        # Study will always refer to study before
        super().__init__(self.study_before, save_to_file=save_to_file, show=not save_to_file)
        self.study_after = study_class(altitude=altitude, year_min=self.year_min_after, year_max=self.year_max_after)
        self.return_period = return_period

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

    def massif_name_to_return_level(self, study):
        massif_name_to_return_levels = OrderedDict()
        for massif_name, maxima in study.massif_name_to_annual_maxima.items():
            gev_param = fitted_stationary_gev(maxima, fit_method=self.fit_method)
            return_level = gev_param.return_level(return_period=self.return_period)
            massif_name_to_return_levels[massif_name] = return_level
        return massif_name_to_return_levels

    def plot_return_level_percentage(self):
        massif_name_to_return_level_past = self.massif_name_to_return_level(self.study_before)
        massif_name_to_return_level_recent = self.massif_name_to_return_level(self.study_after)
        massif_name_to_percentage = {
            m: 100 * (v - massif_name_to_return_level_past[m]) / massif_name_to_return_level_past[m]
            for m, v in
            massif_name_to_return_level_recent.items()}

        max_abs_change = max([abs(e) for e in massif_name_to_percentage.values()])
        ticks, labels = ticks_values_and_labels_for_percentages(graduation=10, max_abs_change=max_abs_change)
        min_ratio = -max_abs_change
        max_ratio = max_abs_change
        cmap = get_shifted_map(min_ratio, max_ratio)
        ax = plt.gca()
        massif_name_to_color = {m: get_colors([v], cmap, min_ratio, max_ratio)[0]
                                for m, v in massif_name_to_percentage.items()}
        ticks_values_and_labels = ticks, labels
        label = 'Relative change in {} return level between \n' \
                'a GEV fitted on {}-{} and a GEV fitted on {}-{} (%)'.format(self.return_period,
                                                                             self.year_min_before,
                                                                             self.year_max_before,
                                                                             self.year_min_after,
                                                                             self.year_max_after)
        AbstractStudy.visualize_study(ax=ax,
                                      massif_name_to_value=massif_name_to_percentage,
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
        self.plot_name = 'change in return levels'
        self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True,
                                  dpi=500)
        plt.close()


if __name__ == '__main__':
    for altitude in [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][:]:
        study_visualizer = StudyVisualizerForReturnLevelChange(study_class=SafranSnowfall1Day, altitude=altitude,
                                                               return_period=30,
                                                               save_to_file=True,
                                                               fit_method=TemporalMarginFitMethod.extremes_fevd_l_moments)
        study_visualizer.plot_return_level_percentage()

