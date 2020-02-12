from typing import Dict
import matplotlib.pyplot as plt

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from papers.exceeding_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends



def get_tuple_ordered_by_shape(fast=False):
    if fast:
        altitudes = [300]
    else:
        altitudes = ALL_ALTITUDES_WITHOUT_NAN
    altitude_to_visualizer = {altitude: StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude),
                                                                              select_only_acceptable_shape_parameter=False,
                                                                              multiprocessing=True)
                              for altitude in altitudes}
    # Extract all the values
    l = []
    for a, v in altitude_to_visualizer.items():
        l.extend([(a, v, m, t.unconstrained_estimator_gev_params.shape) for m, t in
                  v.massif_name_to_trend_test_that_minimized_aic.items()])
    # Sort them and keep the highest examples
    l = sorted(l, key=lambda t: t[-1])
    return l


def plot_qqplot_for_time_series_with_worst_shape_parameters(tuple_ordered_by_shape, nb_worst_examples=5):
    l = tuple_ordered_by_shape
    print('Highest examples:')
    for a, v, m, shape in l[-nb_worst_examples:][::-1]:
        print(a, m, shape)
        v.qqplot(m)
    print('Lowest examples:')
    for a, v, m, shape in l[:1]:
        print(a, m, shape)
        v.qqplot(m)


def plot_return_level_for_time_series_with_big_shape_parameters(tuple_ordered_by_shape, nb_worst_examples=5):
    # Extract all the values
    l = tuple_ordered_by_shape
    print('Highest examples:')
    ax = plt.gca()
    ax2 = ax.twinx()
    colors = ['orange', 'red', 'blue', 'green', 'yellow']
    for (a, v, m, shape), color in zip(l[-nb_worst_examples:][::-1], colors):
        print(a, m, shape, color)
        v.return_level_plot(ax, ax2, m, color)
    plt.show()


"""
10 Worst examples:
300 Oisans 1.1070477650581747
300 Mercantour 1.0857026816954518
300 Haut_Var-Haut_Verdon 0.8446498197950775
600 Haut_Var-Haut_Verdon 0.7058940819679821
300 Devoluy 0.6118436479835319


600 Mercantour 0.4580626323761203
300 Vanoise 0.42632782561692767
600 Ubaye 0.41433496140619247
300 Vercors 0.41292383543990946
300 Bauges 0.39291367817905504
"""

"""
for the worst example for -shape

1500 Vercors 0.5697054636978954
3000 Haute-Tarentaise 0.37528087242416885
2700 Chablais 0.3116920569770965
2700 Aravis 0.2950327331813883
1500 Oisans 0.2942533152179413
1200 Chartreuse 0.2925925517975945
1200 Grandes-Rousses 0.2886979734343222
1500 Devoluy 0.2769805270894181
2100 Chablais 0.27558561202487164
2700 Mont-Blanc 0.2712596135797868
"""

if __name__ == '__main__':
    fast = False
    nb = 1 if fast else 5
    tuple_ordered_by_shape = get_tuple_ordered_by_shape(fast=fast)
    plot_return_level_for_time_series_with_big_shape_parameters(tuple_ordered_by_shape, nb_worst_examples=nb)
