from typing import List
import matplotlib.pyplot as plt
import numpy as np

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit


def plot_coherence_curves(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels]):
    folder = 'Coherence'
    visualizer = visualizer_list[0]
    names = visualizer.get_valid_names(massif_names)
    all_valid_names = set.union(*[v.get_valid_names(massif_names) for v in visualizer_list])
    for massif_name in all_valid_names:
        _, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        colors = ['blue', 'green']
        labels = ['Altitudinal model', 'Pointwise model']
        altitudinal_model = [True, False]
        for color, global_label, boolean in list(zip(colors, labels, altitudinal_model))[:]:
            plot_coherence_curve(axes, massif_name, visualizer_list, boolean, color, global_label)
        visualizer.plot_name = '{}/{}'.format(folder, massif_name.replace('_', '-'))
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()


def plot_coherence_curve(axes, massif_name, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels],
                         is_altitudinal, color, global_label):
    x_all_list, values_all_list, labels, all_bound_list = load_all_list(massif_name, visualizer_list, is_altitudinal)
    for i, label in enumerate(labels):
        ax = axes[i]
        # Plot with complete line
        for j, (x_list, value_list) in enumerate(list(zip(x_all_list, values_all_list))):
            value_list_i = value_list[i]
            label_plot = global_label if j == 0 else None
            if is_altitudinal:
                ax.plot(x_list, value_list_i, linestyle='solid', color=color, label=label_plot)
            else:
                ax.plot(x_list, value_list_i, linestyle='None', color=color, label=label_plot, marker='x')
                ax.plot(x_list, value_list_i, linestyle='dotted', color=color)

        # Plot with dotted line
        for x_list_before, value_list_before, x_list_after, value_list_after in zip(x_all_list, values_all_list,
                                                                                    x_all_list[1:],
                                                                                    values_all_list[1:]):
            x_list = [x_list_before[-1], x_list_after[0]]
            value_list_dotted = [value_list_before[i][-1], value_list_after[i][0]]
            ax.plot(x_list, value_list_dotted, linestyle='dotted', color=color)

        # Plot confidence interval
        if i == 3:
            for x_list, bounds in zip(x_all_list, all_bound_list):
                if len(bounds) > 0:
                    lower_bound, upper_bound = bounds
                    ax.fill_between(x_list, lower_bound, upper_bound, color=color, alpha=0.2)

        ax.set_ylabel(label)

        ax.legend(prop={'size': 10})
        if i >= 2:
            ax.set_xlabel('Altitude')


def load_all_list(massif_name, visualizer_list, altitudinal_model=True):
    all_altitudes_list = []
    all_values_list = []
    all_bound_list = []
    for visualizer in visualizer_list:

        if massif_name in visualizer.massif_name_to_one_fold_fit:
            if altitudinal_model:
                min_altitude, *_, max_altitude = visualizer.massif_name_to_massif_altitudes[massif_name]
                one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
                altitudes_list = list(range(min_altitude, max_altitude, 10))
                gev_params_list = [one_fold_fit.get_gev_params(altitude, 2019) for altitude in altitudes_list]
                confidence_interval_values = [one_fold_fit.best_confidence_interval(altitude) for altitude in altitudes_list]
            else:
                assert OneFoldFit.return_period == 100, 'change the call below'
                altitudes_list, study_list_valid = zip(*[(a, s) for a, s in visualizer.studies.altitude_to_study.items()
                                            if massif_name in s.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[0]])
                gev_params_list = [study.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[0][massif_name]
                                   for study in study_list_valid]
                confidence_interval_values = [study.massif_name_to_stationary_gev_params_and_confidence_for_return_level_100[1][massif_name]
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
    labels = ['location parameter', 'scale parameter', 'shape pameter', '{}-year return level'.format(OneFoldFit.return_period)]
    labels = [l + ' in 2019' for l in labels]
    return all_altitudes_list, all_values_list, labels, all_bound_list


