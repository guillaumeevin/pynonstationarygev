from typing import Dict

import matplotlib.pyplot as plt

from projects.contrasting_trends_in_snow_loads.snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues
from projects.ogorman.gorman_figures.figure1.study_visualizer_for_double_stationary_fit import \
    StudyVisualizerForReturnLevelChange


def validation_plot(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    # Plot the mean empirical, the mean parametric and the relative difference between the two
    altitudes = list(altitude_to_visualizer.keys())
    study_visualizer = list(altitude_to_visualizer.values())[0]
    altitude_to_relative_differences = {}
    # Plot map for the repartition of the difference
    for altitude, visualizer in altitude_to_visualizer.items():
        altitude_to_relative_differences[altitude] = plot_relative_difference_map(visualizer)
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500)
    # Shoe plot with respect to the altitude.
    plot_shoe_relative_differences_distribution(altitude_to_relative_differences, altitudes)
    study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500)


def plot_shoe_relative_differences_distribution(altitude_to_relative_differences, altitudes):
    ax = plt.gca()
    width = 150
    ax.boxplot([altitude_to_relative_differences[a] for a in altitudes], positions=altitudes, widths=width)
    ax.set_xlim([min(altitudes) - width, max(altitudes) + width])
    ylabel = 'Relative difference between empirical mean and parametric mean (\%)'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Altitude (m)')
    ax.legend()
    ax.grid()


def plot_relative_difference_map(visualizer: StudyVisualizerForMeanValues):
    label = ' mean annual maxima of {} ({})'.format('', '')
    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_empirical_mean,
                                  label='Empirical' + label)

    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_parametric_mean,
                                  label='Model' + label)
    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_relative_difference_for_mean,
                                  label='Relative difference of' + label, graduation=1)
    return list(visualizer.massif_name_to_relative_difference_for_mean.values())
