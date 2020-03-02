import time

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.exceeding_snow_loads.paper1_old import get_full_altitude_visualizer, FULL_ALTITUDES


def main_fast_spatial_repartition():
    for altitude in FULL_ALTITUDES[-1:]:
        vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                  exact_starting_year=1958)
        vizualiser.save_to_file = False
        vizualiser.visualize_massif_trend_test_one_altitude()


def main_full_spatial_repartition():
    for altitude in FULL_ALTITUDES[:]:
        # Compute for the most likely starting year
        # vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude)
        # vizualiser.visualize_massif_trend_test_one_altitude()
        # Compute the trend for a linear trend
        vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                  exact_starting_year=1958)
        vizualiser.visualize_massif_trend_test_one_altitude()


def main_run():
    # main_full_spatial_repartition()
    main_fast_spatial_repartition()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))