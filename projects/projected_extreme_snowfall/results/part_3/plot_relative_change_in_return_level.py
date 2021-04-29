from typing import List
import matplotlib.pyplot as plt

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


def plot_relative_dynamic(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], param_name_to_climate_coordinates_with_effects,
                          safran_study_class, relative, order,
                          gcm_rcm_couples
                          ):
    ax = plt.gca()
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    is_temp_covariate = visualizer.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate
    massif_name = massif_names[0]
    for v in visualizer_list:
        plot_curve(ax, massif_name, v, relative, is_temp_covariate, order, gcm_rcm_couples)

    xlabel = 'Anomaly of global temperature w.r.t. pre-industrial levels (K)' if is_temp_covariate else "Years"
    ax.set_xlabel(xlabel)
    change = 'Relative change in' if relative is True else ("Value of" if relative is None else "Change in")
    unit = '\%' if relative is True else (visualizer.study.variable_unit if order != GevParams.SHAPE else 'no unit')
    name = 'the {} parameter'.format(GevParams.full_name_from_param_name(order)) if order is not None \
        else '{}-year return levels'.format(OneFoldFit.return_period)
    ylabel = '{} {} ({})'.format(change, name, unit)
    ax.set_ylabel(ylabel)

    ax.legend(ncol=2, prop={'size': 9}, loc='upper left')
    title = ylabel.split('(')[0]
    set_plot_name(param_name_to_climate_coordinates_with_effects, safran_study_class, title, visualizer)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_curve(ax, massif_name, visualizer: AltitudesStudiesVisualizerForNonStationaryModels,
               relative, is_temp_cov, order, gcm_rcm_couples):
    num = 100
    if is_temp_cov:
        x_list = np.linspace(1, 4.5, num=num)
        covariate_before = 1
    else:
        x_list = np.linspace(1951, 2100, num=num)
        covariate_before = 1951
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    print('relative:', relative, 'order:', order)
    print(get_display_name_from_object_type(type(one_fold_fit.best_margin_model)),
          "significant={}".format(one_fold_fit.is_significant))
    if relative is None:
        f = one_fold_fit.get_moment_for_plots
    else:
        f = one_fold_fit.relative_changes_of_moment if relative else one_fold_fit.changes_of_moment
    color = altitude_to_color[visualizer.study.altitude]
    # Plot the main trend
    changes = [f([None], order=order, covariate_before=covariate_before, covariate_after=t)[0] for t in x_list]
    label = '{} m'.format(visualizer.altitude_group.reference_altitude)
    ax.plot(x_list, changes, label=label, color=color, linewidth=4)
    # Plot the sub trend, i.e. for each GCM-RCM couples
    for gcm_rcm_couple in gcm_rcm_couples[:]:
        fake_altitude = gcm_rcm_couple
        changes = [f([fake_altitude], order=order, covariate_before=covariate_before, covariate_after=t)[0] for t in x_list]
        ax.plot(x_list, changes, color=color, linewidth=1, linestyle='dotted')


altitude_to_color = {
    # Low altitude group
    600: 'darkred',
    900: 'red',
    # Mid altitude gruop
    1200: 'darkorange',
    1500: 'orange',
    1800: 'gold',
    # High altitude group
    2100: 'lightskyblue',
    2400: 'skyblue',
    2700: 'dodgerblue',
    # Very high altitude group
    3000: 'b',
    3300: 'mediumblue',
    3600: 'darkblue',
}


def set_plot_name(param_name_to_climate_coordinates_with_effects, safran_study_class, title, visualizer):
    # raise NotImplementedError
    plot_name = ' %s' % title
    plot_name += ' with {} effects'.format('no' if param_name_to_climate_coordinates_with_effects is None
                                           else ' and '.join(
        [c.replace('coord_', '') for c in param_name_to_climate_coordinates_with_effects]))
    plot_name += ' with{} observations'.format('out' if safran_study_class is None else '')
    visualizer.plot_name = plot_name
