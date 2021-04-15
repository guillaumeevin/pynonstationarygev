from typing import List
import matplotlib.pyplot as plt

import numpy as np

from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


def plot_relative_dynamic_in_return_level(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], climate_coordinates_with_effects,
                                          safran_study_class, relative
                                          ):
    ax = plt.gca()
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    assert visualizer.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate
    massif_name = massif_names[0]
    for v in visualizer_list:
        plot_curve(ax, massif_name, v, relative)

    ax.set_xlabel('Anomaly of global temperature w.r.t. pre-industrial levels (K)')
    change = 'Relative change' if relative else "Change"
    unit = '\%' if relative else visualizer.study.variable_unit
    ax.set_ylabel((change + (' in return levels (' + unit + ')')))

    ax.legend(ncol=2, prop={'size': 9}, loc='upper left')
    title = 'change {}of return level'.format('relative ' if relative else '')
    set_plot_name(climate_coordinates_with_effects, safran_study_class, title, visualizer)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()

def set_plot_name(climate_coordinates_with_effects, safran_study_class, title, visualizer):
    plot_name = ' %s' % title
    plot_name += ' with {} effects'.format('no' if climate_coordinates_with_effects is None
                                         else ' and '.join(
        [c.replace('coord_', '') for c in climate_coordinates_with_effects]))
    plot_name += ' with{} observations'.format('out' if safran_study_class is None else '')
    visualizer.plot_name = plot_name


def plot_curve(ax, massif_name, visualizer: AltitudesStudiesVisualizerForNonStationaryModels,
               relative):
    temperatures_list = np.linspace(1, 4.5, num=400)
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    print(get_display_name_from_object_type(type(one_fold_fit.best_margin_model)),
          "significant={}".format(one_fold_fit.is_significant))
    f = one_fold_fit.relative_changes_of_moment if relative else one_fold_fit.changes_of_moment
    return_levels = [f([None], order=None, covariate_before=1, covariate_after=t)[0] for t in temperatures_list]
    label = '{} m'.format(visualizer.altitude_group.reference_altitude)
    ax.plot(temperatures_list, return_levels, label=label)
