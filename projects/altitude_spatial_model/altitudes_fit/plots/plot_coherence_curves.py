from typing import List
import matplotlib.pyplot as plt

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
        plot_coherence_curve(massif_name, visualizer_list)
        visualizer.plot_name = '{}/{}'.format(folder, massif_name.replace('_', '-'))
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()


def plot_coherence_curve(massif_name, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels]):
    x_all_list, values_all_list, labels = load_all_list(massif_name, visualizer_list)
    _, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for i, label in enumerate(labels):
        ax = axes[i]
        # Plot with complete line
        for x_list, value_list in zip(x_all_list, values_all_list):
            value_list = value_list[i]
            ax.plot(x_list, value_list, linestyle='solid')
        # Plot with dotted line
        for x_list_before, value_list_before, x_list_after, value_list_after in zip(x_all_list, values_all_list,
                                                                                    x_all_list[1:],
                                                                                    values_all_list[1:]):
            x_list = [x_list_before[-1], x_list_after[0]]
            value_list = [value_list_before[i][-1], value_list_after[i][0]]
            ax.plot(x_list, value_list, linestyle='dotted')

        ax.set_ylabel(label)
        ax.set_xlabel('Altitude')


def load_all_list(massif_name, visualizer_list):
    all_altitudes_list = []
    all_values_list = []
    for visualizer in visualizer_list:
        if massif_name in visualizer.massif_name_to_one_fold_fit:
            min_altitude, *_, max_altitude = visualizer.massif_name_to_massif_altitudes[massif_name]
            altitudes_list = list(range(min_altitude, max_altitude, 10))
            gev_params_list = []
            for altitude in altitudes_list:
                gev_params = visualizer.massif_name_to_one_fold_fit[massif_name].get_gev_params(altitude, 2019)
                gev_params_list.append(gev_params)
            values = [(gev_params.location, gev_params.scale, gev_params.shape,
                       gev_params.return_level(return_period=OneFoldFit.return_period))
                      for gev_params in gev_params_list]
            values_list = list(zip(*values))
            all_values_list.append(values_list)
            all_altitudes_list.append(altitudes_list)
    labels = ['loc', 'scale', 'shape', 'return level']
    return all_altitudes_list, all_values_list, labels
