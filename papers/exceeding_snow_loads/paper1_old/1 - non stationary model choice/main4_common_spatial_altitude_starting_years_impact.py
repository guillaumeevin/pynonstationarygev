import time

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer, AltitudeHypercubeVisualizerWithoutTrendType
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest

"""
Visualize the 0.99 quantile initial value and its evolution
"""
from experiment.exceeding_snow_loads.paper1_old import get_full_altitude_visualizer, FULL_ALTITUDES


def main_fast_spatial_risk_evolution():
    vizualiser = get_full_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendType, altitude=None,
                                              reduce_strength_array=True,
                                              trend_test_class=GevLocationAndScaleTrendTest,
                                              offset_starting_year=28)
    vizualiser.save_to_file = False
    vizualiser.sigma_for_best_year = 1.0
    res = vizualiser.visualize_year_trend_test(subtitle_specified='CrocusSwe3Days')
    print(res)


def main_full_spatial_risk_evolution():
    for trend_test_class in [GevLocationAndScaleTrendTest]:
        # Compare the risk with and without taking into account the starting year
        vizualiser = get_full_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendType, altitude=None,
                                                  reduce_strength_array=True,
                                                  trend_test_class=trend_test_class,
                                                  offset_starting_year=20)
        vizualiser.sigma_for_best_year = 1.0
        res = vizualiser.visualize_year_trend_test(subtitle_specified='CrocusSwe3Days')
        best_year = res[0][1]
        for altitude in FULL_ALTITUDES[:]:
            # Starting Year=1958
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=True,
                                                      trend_test_class=trend_test_class)
            vizualiser.visualize_massif_trend_test_one_altitude()
            # Optimal common starting year
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=best_year, reduce_strength_array=True,
                                                      trend_test_class=trend_test_class)
            vizualiser.visualize_massif_trend_test_one_altitude()


def main_run():
    main_full_spatial_risk_evolution()
    # main_fast_spatial_risk_evolution()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
