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
    x, gev_params_list = load_all_list(massif_name, visualizer_list)
    values = [(gev_params.location, gev_params.scale, gev_params.shape, gev_params.return_level(return_period=OneFoldFit.return_period))
              for gev_params in gev_params_list]
    values = list(zip(*values))
    labels = ['loc', 'scale', 'shape', 'return level']
    _, axes = plt.subplots(2, 2)
    for label, value, ax in zip(labels, values, axes.flatten()):
        ax.plot(x, value)
        ax.set_ylabel(label)
        ax.set_xlabel('Altitude')

def load_all_list(massif_name, visualizer_list):
    all_altitudes_list = []
    all_gev_params_list = []
    for visualizer in visualizer_list:
        if massif_name in visualizer.massif_name_to_one_fold_fit:
            min_altitude, *_, max_altitude = visualizer.massif_name_to_massif_altitudes[massif_name]
            altitudes_list = list(range(min_altitude, max_altitude, 10))
            gev_params_list = []
            for altitude in altitudes_list:
                gev_params = visualizer.massif_name_to_one_fold_fit[massif_name].get_gev_params(altitude, 2019)
                gev_params_list.append(gev_params)
            all_gev_params_list.extend(gev_params_list)
            all_altitudes_list.extend(altitudes_list)
    return all_altitudes_list, all_gev_params_list
