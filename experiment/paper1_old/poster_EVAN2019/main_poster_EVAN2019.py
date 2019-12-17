from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer, AltitudeHypercubeVisualizerWithoutTrendType
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDIES
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter import GevScaleTrendTest, \
    GevLocationTrendTest
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from experiment.paper1_old.utils import get_full_altitude_visualizer

POSTER_ALTITUDES = [900, 1800, 2700]
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.3


def main_poster_A_non_stationary_model_choice():
    nb = 1
    for altitude in POSTER_ALTITUDES[:]:
        for trend_test_class in [GevLocationTrendTest, GevScaleTrendTest, GevLocationAndScaleTrendTest][-nb:]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      )
            # vizualiser.save_to_file = False
            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)


def main_poster_B_starting_years_analysis():
    nb = 3
    for altitude in POSTER_ALTITUDES[:nb]:
        for trend_test_class in [GevLocationAndScaleTrendTest]:
            # 1958 as starting year
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      )
            for d in [True, False]:
                vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
                                                                    display_trend_color=d)
            # vizualiser.save_to_file = False

            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)
            # Optimal common starting year
            vizualiser = get_full_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendType, altitude=altitude,
                                                      reduce_strength_array=True,
                                                      trend_test_class=trend_test_class,
                                                      offset_starting_year=20)
            res = vizualiser.visualize_year_trend_test(subtitle_specified='CrocusSwe3Days')
            best_year = res[0][1]
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=best_year, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class)
            for d in [True, False]:
                vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
                                                                    display_trend_color=d)
            # Individual most likely starting year for each massif
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      offset_starting_year=20)
            for d in [True, False]:
                vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
                                                                    display_trend_color=d)

# def main_poster_B_test():
#     nb = 3
#     for altitude in POSTER_ALTITUDES[:1]:
#         for trend_test_class in [GevLocationAndScaleTrendTest]:
#             # # 1958 as starting year
#             vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
#                                                       exact_starting_year=1958, reduce_strength_array=False,
#                                                       trend_test_class=trend_test_class,
#                                                       )
#             # vizualiser.save_to_file = False
#             vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
#                                                                 display_trend_color=False)
#             vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
#                                                                 display_trend_color=True)
#             # # Optimal common starting year
#             vizualiser = get_full_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendType, altitude=altitude,
#                                                       reduce_strength_array=True,
#                                                       trend_test_class=trend_test_class,
#                                                       offset_starting_year=20)
#             res = vizualiser.visualize_year_trend_test(subtitle_specified='CrocusSwe3Days')
#             best_year = res[0][1]
#             vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
#                                                       exact_starting_year=best_year, reduce_strength_array=False,
#                                                       trend_test_class=trend_test_class)
#             vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
#                                                                 display_trend_color=False)
#             vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False,
#                                                                 display_trend_color=True)
#             # vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)
#             # Individual most likely starting year for each massif
#             # vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
#             #                                           reduce_strength_array=False,
#             #                                           trend_test_class=trend_test_class,
#             #                                           offset_starting_year=50)
#             # # vizualiser.save_to_file = False
#             # vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=True,
#             #                                                     display_trend_color=False)



def main_poster_C_orientation_analysis():
    """By default the slope is equal to 20"""
    nb = 0
    cardinal_orientations = [0.0, 90.0, 180.0, 270.0]
    trend_test_class = GevLocationAndScaleTrendTest
    for altitude in POSTER_ALTITUDES[nb:]:
        study_class = CrocusSwe3Days
        for orientation in cardinal_orientations[nb:]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      study_class=study_class,
                                                      orientation=orientation)
            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)


def main_poster_D_other_quantities_analysis():
    nb = 3
    trend_test_class = GevLocationAndScaleTrendTest
    for altitude in POSTER_ALTITUDES[:nb]:
        for study_class in SCM_STUDIES[:nb]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      study_class=study_class)
            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)


if __name__ == '__main__':
    main_poster_A_non_stationary_model_choice()
    # main_poster_B_starting_years_analysis()
    # main_poster_C_orientation_analysis()
    # main_poster_D_other_quantities_analysis()
