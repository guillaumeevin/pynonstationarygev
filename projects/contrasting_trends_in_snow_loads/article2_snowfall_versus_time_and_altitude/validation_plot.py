from typing import Dict

import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues
from projects.ogorman.gorman_figures.figure1.study_visualizer_for_double_stationary_fit import \
    StudyVisualizerForReturnLevelChange


def validation_plot(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues], order_derivative=0):
    # Plot the mean empirical, the mean parametric and the relative difference between the two
    altitudes = list(altitude_to_visualizer.keys())
    study_visualizer = list(altitude_to_visualizer.values())[0]
    altitude_to_relative_differences = {}
    if order_derivative == 0:
        plot_function = plot_relative_difference_map_order_zero
    else:
        plot_function = plot_relative_difference_map_order_one
    # Plot map for the repartition of the difference
    for altitude, visualizer in altitude_to_visualizer.items():
        altitude_to_relative_differences[altitude] = plot_function(visualizer)
        study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500)
    # # Shoe plot with respect to the altitude.
    # plot_shoe_relative_differences_distribution(altitude_to_relative_differences, altitudes, study_visualizer,
    #                                             order_derivative)
    study_visualizer.show_or_save_to_file(add_classic_title=False, dpi=500)
    plt.close()


def plot_shoe_relative_differences_distribution(altitude_to_relative_differences, altitudes, visualizer,
                                                order_derivative):
    study = visualizer.study
    ax = plt.gca()
    width = 150
    ax.boxplot([altitude_to_relative_differences[a] for a in altitudes], positions=altitudes, widths=width)
    ax.set_xlim([min(altitudes) - width, max(altitudes) + width])
    moment = '' if order_derivative == 0 else 'time derivative of '
    ylabel = 'Global relative difference of the {} model mean \n' \
             'w.r.t. the {}empirical mean of {} (\%)'.format(moment, moment,
                                                             SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)])
    visualizer.plot_trends = ylabel
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Altitude (m)')
    ax.legend()
    ax.grid()


def plot_relative_difference_map_order_zero(visualizer: StudyVisualizerForMeanValues):
    study = visualizer.study
    label = ' mean annual maxima of {} ({})'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)], study.variable_unit)
    # visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_empirical_mean,
    #                               label='Empirical' + label, negative_and_positive_values=False)
    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_model_mean,
                                  label='Model' + label, negative_and_positive_values=False, add_text=True)
    # visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_relative_difference_for_mean,
    #                               label='Relative difference of the model mean w.r.t. the empirical mean \n'
    #                                     'for the ' + label, graduation=1)
    return list(visualizer.massif_name_to_relative_difference_for_mean.values())


def plot_relative_difference_map_order_one(visualizer: StudyVisualizerForMeanValues):
    study = visualizer.study
    label = ' time derivative of mean annual maxima of {} ({})'.format(SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)],
                                                                       study.variable_unit)
    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_change_ratio_in_empirical_mean,
                                  label='Empirical' + label, negative_and_positive_values=False, graduation=0.5)
    visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_change_ratio_in_model_mean,
                                  label='Model' + label, negative_and_positive_values=False, graduation=0.5)
    visualizer.plot_abstract_fast(
        massif_name_to_value=visualizer.massif_name_to_relative_difference_for_change_ratio_in_mean,
        label='Relative difference of the model mean w.r.t. the empirical mean \n'
              'for the ' + label, graduation=5)
    return list(visualizer.massif_name_to_relative_difference_for_change_ratio_in_mean.values())
