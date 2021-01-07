from typing import List
import matplotlib.pyplot as plt
import numpy as np

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit


def plot_coherence_curves(massif_names, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels]):
    folder = 'Coherence'
    visualizer = visualizer_list[0]
    all_valid_names = set.union(*[v.get_valid_names(massif_names) for v in visualizer_list])
    for massif_name in all_valid_names:

        # For plotting the legend
        legend = False
        if legend:
            ax = plt.gca()
            axes = [ax for _ in range(4)]
        else:
            _, axes = plt.subplots(2, 2)
            axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i % 2 == 1:
                ax.set_yticks([])
        axes = [ax if i % 2 == 0 else ax.twinx() for i, ax in enumerate(axes)]
        colors = ['blue', 'red', 'green']
        elevational_str = 'Piecewise elevational-temporal models in'
        labels = ['{} 1959'.format(elevational_str), '{} 2019'.format(elevational_str), 'Pointwise distributions']
        altitudinal_model = [True, True, False]
        years = [1959, 2019, None]
        for color, global_label, boolean, year in list(zip(colors, labels, altitudinal_model, years))[:]:
            plot_coherence_curve(axes, massif_name, visualizer_list, boolean, color, global_label, year, legend)
        visualizer.plot_name = '{}/{}'.format(folder, massif_name.replace('_', '-'))
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True, dpi=200)
        plt.close()


def plot_coherence_curve(axes, massif_name, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels],
                         is_altitudinal, color, global_label, year, legend):
    x_all_list, values_all_list, labels, all_bound_list = load_all_list(massif_name, visualizer_list, is_altitudinal,
                                                                        year)

    legend_line = True
    for i, label in enumerate(labels):
        if legend and i != 3:
            continue
        ax = axes[i]
        # Plot with complete line
        for j, (x_list, value_list) in enumerate(list(zip(x_all_list, values_all_list))):
            value_list_i = value_list[i]
            label_plot = global_label if j == 0 else None
            if is_altitudinal:
                if legend and legend_line:
                    ax.plot(x_list, value_list_i, linestyle='solid', color=color, label=label_plot, linewidth=5)
                else:
                    ax.plot(x_list, value_list_i, linestyle='solid', color=color)
            else:
                if legend and legend_line:
                    ax.plot(x_list, value_list_i, linestyle='None', color=color, label=label_plot, marker='o', markersize=10)
                else:
                    ax.plot(x_list, value_list_i, linestyle='None', color=color, marker='o')
                    ax.plot(x_list, value_list_i, linestyle='dotted', color=color)

        # Plot with dotted line
        for x_list_before, value_list_before, x_list_after, value_list_after in zip(x_all_list, values_all_list,
                                                                                    x_all_list[1:],
                                                                                    values_all_list[1:]):
            x_list = [x_list_before[-1], x_list_after[0]]
            value_list_dotted = [value_list_before[i][-1], value_list_after[i][0]]
            ax.plot(x_list, value_list_dotted, linestyle='dotted', color=color)

        # Plot confidence interval
        if i == 3 and year in [None, 2019]:
            for j, (x_list, bounds) in enumerate(list(zip(x_all_list, all_bound_list))):
                if len(bounds) > 0:
                    lower_bound, upper_bound = bounds
                    if legend and not legend_line:
                        model_name = 'piecewise elevational-temporal models in 2019' if is_altitudinal else 'pointwise distributions'
                        fill_label = "95\% confidence interval for the {}".format(model_name) if j == 0 else None
                        ax.fill_between(x_list, lower_bound, upper_bound, color=color, alpha=0.2, label=fill_label)
                    else:
                        ax.fill_between(x_list, lower_bound, upper_bound, color=color, alpha=0.2)

            if legend:
                min, max = ax.get_ylim()
                ax.set_ylim([min, 2 * max])
                size = 15 if legend_line else 11
                ax.legend(prop={'size': size})
        ax.set_ylabel(label)


def load_all_list(massif_name, visualizer_list, altitudinal_model=True, year=2019):
    all_altitudes_list = []
    all_values_list = []
    all_bound_list = []
    for visualizer in visualizer_list:

        if massif_name in visualizer.massif_name_to_one_fold_fit:
            if altitudinal_model:
                min_altitude, *_, max_altitude = visualizer.massif_name_to_massif_altitudes[massif_name]
                one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
                altitudes_list = list(range(min_altitude, max_altitude, 10))
                gev_params_list = [one_fold_fit.get_gev_params(altitude, year) for altitude in altitudes_list]
                confidence_interval_values = [one_fold_fit.best_confidence_interval(altitude, year) for altitude in
                                              altitudes_list]
            else:
                assert OneFoldFit.return_period == 100, 'change the call below'
                altitudes_list, study_list_valid = zip(*[(a, s) for a, s in visualizer.studies.altitude_to_study.items()
                                                         if massif_name in
                                                         s.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[
                                                             0]])
                gev_params_list = [
                    study.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[0][massif_name]
                    for study in study_list_valid]
                confidence_interval_values = [
                    study.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[1][massif_name]
                    for study in study_list_valid]

            # Checks
            values = [(gev_params.location, gev_params.scale, gev_params.shape,
                       gev_params.return_level(return_period=OneFoldFit.return_period))
                      for gev_params in gev_params_list]
            for a, b in zip(values, confidence_interval_values):
                if not np.isnan(b.mean_estimate):
                    assert np.isclose(a[-1], b.mean_estimate)
            bound_list = [c.confidence_interval for c in confidence_interval_values if not np.isnan(c.mean_estimate)]
            values_list = list(zip(*values))
            all_values_list.append(values_list)
            all_altitudes_list.append(altitudes_list)
            all_bound_list.append(list(zip(*bound_list)))
    labels = ['location', 'scale', 'shape', '{}-year return levels'.format(OneFoldFit.return_period)]

    for i, label in enumerate(labels):
        if i < 3:
            label += ' parameter'
        unit = 'no unit' if i == 2 else visualizer_list[0].study.variable_unit
        label += ' ({})'.format(unit)
        labels[i] = label

    return all_altitudes_list, all_values_list, labels, all_bound_list
