from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSweTotal, CrocusSwe3Days
from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer, AltitudeHypercubeVisualizerWithoutTrendType
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDIES
from experiment.trend_analysis.univariate_test.abstract_comparison_non_stationary_model import ComparisonAgainstMu, \
    ComparisonAgainstSigma
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevScaleTrendTest, \
    GevLocationTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from experiment.paper1_steps.utils import get_full_altitude_visualizer

POSTER_ALTITUDES = [900, 1800, 2700]
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.3


def main_non_stationary_model_comparison():
    stop_loop = False
    for altitude in POSTER_ALTITUDES[:]:
        for trend_test_class in [GevLocationTrendTest, GevScaleTrendTest, GevLocationAndScaleTrendTest,
                                 ComparisonAgainstMu, ComparisonAgainstSigma][:]:
            vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                      exact_starting_year=1958, reduce_strength_array=False,
                                                      trend_test_class=trend_test_class,
                                                      verbose=False)
            # vizualiser.save_to_file = False
            vizualiser.visualize_massif_trend_test_one_altitude(poster_plot=True, write_text_on_massif=False)
            if stop_loop:
                return


if __name__ == '__main__':
    main_non_stationary_model_comparison()
