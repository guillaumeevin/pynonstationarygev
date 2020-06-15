from typing import Dict

import matplotlib
import matplotlib.pyplot as plt

from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues


def shape_plot(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    # Plot map for the repartition of the difference
    for altitude, visualizer in altitude_to_visualizer.items():
        label = ' shape parameter'
        visualizer.plot_abstract_fast(massif_name_to_value=visualizer.massif_name_to_model_shape_last_year,
                                      label='Model' + label, negative_and_positive_values=True, add_text=True,
                                      cmap=matplotlib.cm.get_cmap('BrBG_r'), graduation=0.1)
    plt.close()
