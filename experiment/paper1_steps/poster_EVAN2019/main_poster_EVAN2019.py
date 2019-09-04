from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevScaleTrendTest, \
    GevLocationTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from experiment.paper1_steps.utils import get_full_altitude_visualizer

POSTER_ALTITUDES = [900, 1800, 2700]


def main_poster_A_non_stationary_model_choice():
    nb = 1
    for altitude in POSTER_ALTITUDES[:nb]:
        for trend_test_class in [GevLocationTrendTest, GevScaleTrendTest, GevLocationAndScaleTrendTest][-nb:]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      )
            # vizualiser.save_to_file = False
            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)


def main_poster_B_starting_years_analysis():
    pass


if __name__ == '__main__':
    main_poster_A_non_stationary_model_choice()
