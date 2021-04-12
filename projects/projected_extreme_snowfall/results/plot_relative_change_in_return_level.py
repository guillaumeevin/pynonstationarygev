from typing import List
import matplotlib.pyplot as plt

import numpy as np

from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


def plot_relative_dynamic_in_return_level(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels]):
    ax = plt.gca()
    print(len(visualizer_list))
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    assert visualizer.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate
    massif_name = massif_names[0]
    for v in visualizer_list:
        plot_curve(ax, massif_name, v)

    ax.set_xlabel('Anomaly of global temperature w.r.t. pre-industrial levels (K)')
    ax.set_ylabel('Relative change in return levels (\%)')

    ax.legend()
    visualizer.plot_name = 'dynamic of return level'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_curve(ax, massif_name, visualizer: AltitudesStudiesVisualizerForNonStationaryModels):
    temperatures_list = np.linspace(1, 5, num=40)
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    print(get_display_name_from_object_type(type(one_fold_fit.best_margin_model)),
          "significant={}".format(one_fold_fit.is_significant))
    return_levels = [one_fold_fit.relative_changes_of_moment([None], order=None,
                                                             covariate_before=1,
                                                             covariate_after=t)[0] for t in temperatures_list]
    label = '{} m'.format(visualizer.altitude_group.reference_altitude)
    ax.plot(temperatures_list, return_levels, label=label)


