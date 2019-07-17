import time

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest, \
    GevScaleChangePointTest

"""
Visualize the 0.99 quantile initial value and its evolution
"""
from experiment.paper1_steps.utils import get_full_altitude_visualizer, FULL_ALTITUDES


def main_fast_spatial_risk_evolution():
    for altitude in FULL_ALTITUDES[-1:]:
        vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                  exact_starting_year=1958, reduce_strength_array=True,
                                                  trend_test_class=GevScaleChangePointTest)
        # vizualiser.save_to_file = False
        vizualiser.visualize_massif_trend_test_one_altitude()


def main_full_spatial_risk_evolution():
    for altitude in FULL_ALTITUDES[:]:
        for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest][:]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=True,
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
