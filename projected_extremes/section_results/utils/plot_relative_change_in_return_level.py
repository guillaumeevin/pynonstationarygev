from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
                          gcm_rcm_couples, with_significance
                          ):
    ax = plt.gca()
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    is_temp_covariate = visualizer.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate
    massif_name = massif_names[0]
    for v in visualizer_list:
        plot_curve(ax, massif_name, v, relative, is_temp_covariate, order, gcm_rcm_couples, with_significance)

    xlabel = 'T, the smoothed anomaly of global temperature w.r.t. pre-industrial levels (K)' if is_temp_covariate else "Years"
    ax.set_xlabel(xlabel)
    change = 'Relative change in' if relative is True else ("" if relative is None else "Change in")
    unit = '\%' if relative is True else (visualizer.study.variable_unit if order != GevParams.SHAPE else 'no unit')
    if order is None:
        name = '{}-year return levels'.format(OneFoldFit.return_period)
    elif order is True:
        name = "Mean annual maxima"
    else:
        name = '{} parameter'.format(GevParams.full_name_from_param_name(order))
    ylabel = '{} {} ({})'.format(change, name, unit)
    ylabel = ylabel.strip()
    ylabel = ylabel[0].upper() + ylabel[1:]
    ax.set_ylabel(ylabel)

    h, l = ax.get_legend_handles_labels()
    if len(h) > 1:
        ax.legend(prop={'size': 14})
    title = '{} {} for the {} massif'.format(change, name, massif_name)
    set_plot_name(param_name_to_climate_coordinates_with_effects, safran_study_class, title, visualizer, massif_name)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_curve(ax, massif_name, visualizer: AltitudesStudiesVisualizerForNonStationaryModels,
               relative, is_temp_cov, order, gcm_rcm_couples,
               with_significance):
    q_list = [0.05, 0.95]
    width = 100 * (q_list[1] - q_list[0])
    alpha = 0.1

    num = 100
    if is_temp_cov:
        x_list = np.linspace(1, 4.5, num=num)
        covariate_before = 1
    else:
        x_list = np.linspace(1951, 2100, num=num)
        covariate_before = 1951
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    print('\nplot curve')
    print(visualizer.study.altitude)
    print('relative:', relative, 'order:', order)
    print(get_display_name_from_object_type(type(one_fold_fit.best_margin_model)),
          "temporally significant={}".format(one_fold_fit.is_significant))
    if one_fold_fit.param_name_to_climate_coordinates_with_effects is not None:
        print("all effects significant={}".format(one_fold_fit.correction_is_significant))
    # print("gcm effects significant={}".format(one_fold_fit.gcm_correction_is_significant))
    # print("rcm effects significant={}".format(one_fold_fit.rcm_correction_is_significant))
    if relative is None:
        f = one_fold_fit.get_moment_for_plots
    else:
        f = one_fold_fit.relative_changes_of_moment if relative else one_fold_fit.changes_of_moment
    altitude = visualizer.study.altitude
    color = altitude_to_color[altitude]

    snowfall = False

    # Plot the sub trend, i.e. for each GCM-RCM couples
    # for gcm_rcm_couple in gcm_rcm_couples[:]:
    #     fake_altitude = gcm_rcm_couple
    #     gcm_rcm_color = color if snowfall else gcm_rcm_couple_to_color[gcm_rcm_couple]
    #     changes = [f([fake_altitude], order=order, covariate_before=covariate_before, covariate_after=t)[0] for t in
    #                x_list]
    #     ax.plot(x_list, changes, color=gcm_rcm_color, linewidth=1, linestyle='dotted')

    # Plot the main trend
    changes = [f([None], order=order, covariate_before=covariate_before, covariate_after=t)[0] for t in x_list]
    label = '{} m'.format(visualizer.altitude_group.reference_altitude)
    obs_color = color if snowfall else 'k'
    ax.plot(x_list, changes, label=label, color=obs_color, linewidth=2)

    # Additional plots for the value of return level
    if relative is None:
        # Plot the uncertainty interval
        if with_significance:
            margin_functions = one_fold_fit.bootstrap_fitted_functions_from_fit_cached
            coordinates_list = [np.array([t]) for t in x_list]

            if order is None:
                values = [[f.get_params(c).return_level(OneFoldFit.return_period) for c in coordinates_list] for
                                 f in
                                 margin_functions]
            elif order is True:
                if order is None:
                    values = [[f.get_params(c).mean for c in coordinates_list] for
                              f in
                              margin_functions]
            else:
                values = [[f.get_params(c).to_dict()[order] for c in coordinates_list] for
                                 f in
                                 margin_functions]

            lower_bound, upper_bound = [np.quantile(values, q, axis=0) for q in q_list]
            ax.fill_between(x_list, lower_bound, upper_bound, color=obs_color, alpha=alpha)

        # Plot the structure standard as reference for the snow load
        # if not snowfall:
        #     eurocode_region = massif_name_to_eurocode_region[massif_name]()
        #     constant_norm = eurocode_region.valeur_caracteristique(altitude)
        #     ax.plot(x_list, [constant_norm for _ in x_list], color=obs_color, linestyle='dashed')

        if order is None:
            label_global = "$\\textrm{RL50}"
        elif order is True:
            label_global = '$\\textrm{Mean annual maxima}'
        else:
            label_global = "$\\" + GevParams.greek_letter_from_param_name_confidence_interval(order, linearity_in_shape=True)

        label_obs = label_global + '(\\textrm{T})$'

        ax2 = ax.twinx()
        legend_elements = [
            Line2D([0], [0], color='k', lw=3, label=label_obs, linestyle='-'),
            # Line2D([0], [0], color='k', lw=1, label="French building standard", linestyle='dashed'),
        ]
        # if order != GevParams.SHAPE:
        #     for gcm, color in gcm_to_color.items():
        #         label_gcm = label_global + '(\\textrm{T,' + gcm + '},\\cdot)$'
        #         legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label_gcm, linestyle='dotted'))
        legend_elements.append(Patch(facecolor='k', edgecolor='k', label="90\% uncertainty interval for {}".format(label_obs), alpha=alpha),)

        size = 9
        loc = 'upper left' if order is GevParams.SHAPE else 'upper right'
        ax2.legend(handles=legend_elements, loc=loc, prop={'size': size}, handlelength=3)
        ax2.set_yticks([])


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


def set_plot_name(param_name_to_climate_coordinates_with_effects, safran_study_class, title, visualizer, massif_name):
    # raise NotImplementedError
    plot_name = ' %s' % title
    plot_name += ' with {} effects'.format('no' if param_name_to_climate_coordinates_with_effects is None
                                           else ' and '.join(
        [c.replace('coord_', '') for c in param_name_to_climate_coordinates_with_effects]))
    plot_name += ' with{} observations'.format('out' if safran_study_class is None else '')
    plot_name += massif_name.replace('-', '_')
    visualizer.plot_name = plot_name
