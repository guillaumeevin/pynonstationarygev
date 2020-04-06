from collections import OrderedDict

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, shiftedColorMap
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod

import matplotlib.pyplot as plt

from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import StudyVisualizerForNonStationaryTrends


def mean_ratio(altitude):
    pass


def massif_name_to_return_level(altitude, year_min, year_max, return_period):
    study = SafranSnowfall1Day(altitude=altitude, year_min=year_min, year_max=year_max)
    massif_name_to_return_levels = OrderedDict()
    for massif_name, maxima in study.massif_name_to_annual_maxima.items():
        gev_param = fitted_stationary_gev(maxima, fit_method=TemporalMarginFitMethod.extremes_fevd_mle)
        return_level = gev_param.return_level(return_period=return_period)
        massif_name_to_return_levels[massif_name] = return_level
    return massif_name_to_return_levels


def plot_return_level_ratio(altitude, return_period=30, year_min=1959, year_middle=1989, year_max=2019):
    massif_name_to_return_level_past = massif_name_to_return_level(altitude, year_min, year_middle, return_period)
    massif_name_to_return_level_recent = massif_name_to_return_level(altitude, year_middle + 1, year_max,
                                                                     return_period)
    massif_name_to_ratio = {m: v / massif_name_to_return_level_past[m] for m, v in
                            massif_name_to_return_level_recent.items()}
    max_ratio = max([e for e in massif_name_to_ratio.values()])
    min_ratio = min([e for e in massif_name_to_ratio.values()])
    midpoint = (max_ratio - 1.0) / (max_ratio - 0)
    num = 3
    ticks = np.concatenate([np.linspace(0, midpoint, num), np.linspace(midpoint, 1, num)])
    labels = np.concatenate([np.linspace(min_ratio, 1.0, num), np.linspace(1.0, max_ratio, num)])
    cmap = shiftedColorMap(plt.cm.bwr, midpoint=midpoint, name='shifted')
    ax = plt.gca()
    print(massif_name_to_ratio)
    massif_name_to_color = {m: get_colors([v], cmap, min_ratio, max_ratio)[0]
                for m, v in massif_name_to_ratio.items()}
    ticks_values_and_labels = ticks, labels
    AbstractStudy.visualize_study(ax=ax,
                                  massif_name_to_value=massif_name_to_ratio,
                                  massif_name_to_color=massif_name_to_color,
                                  replace_blue_by_white=True,
                                  axis_off=False,
                                  cmap=cmap,
                                  show_label=False,
                                  add_colorbar=True,
                                  show=False,
                                  vmin=min_ratio,
                                  vmax=max_ratio,
                                  ticks_values_and_labels=ticks_values_and_labels
                                  )
    plt.show()


if __name__ == '__main__':
    plot_return_level_ratio(altitude=1800)
