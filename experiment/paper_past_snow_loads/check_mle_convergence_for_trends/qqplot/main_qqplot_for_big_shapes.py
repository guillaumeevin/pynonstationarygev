from typing import Dict

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


def qqplots_for_biggest_shape_parameters_before_selection():
    altitudes = ALL_ALTITUDES_WITHOUT_NAN
    altitude_to_visualizer = {altitude: StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude),
                                                                              select_only_acceptable_shape_parameter=False,
                                                                              multiprocessing=True)
                              for altitude in altitudes}
    plot_qqplot_for_time_series_with_worst_shape_parameters(altitude_to_visualizer, nb_worst_examples=5)


def plot_qqplot_for_time_series_with_worst_shape_parameters(
        altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
        nb_worst_examples=3):
    # Extract all the values
    l = []
    for a, v in altitude_to_visualizer.items():
        l.extend([(a, v, m, t.unconstrained_estimator_gev_params.shape) for m, t in v.massif_name_to_trend_test_that_minimized_aic.items()])
    # Sort them and keep the highest examples
    l = sorted(l, key=lambda t: t[-1])
    print('Highest examples:')
    for a, v, m, shape in l[-nb_worst_examples:][::-1]:
        print(a, m, shape)
        v.qqplot(m)
    print('Lowest examples:')
    for a, v, m, shape in l[:1]:
        print(a, m, shape)
        v.qqplot(m)

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
    qqplots_for_biggest_shape_parameters_before_selection()
