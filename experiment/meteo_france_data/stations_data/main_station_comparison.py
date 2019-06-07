from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST
from experiment.meteo_france_data.stations_data.visualization.comparisons_visualization.comparisons_visualization import \
    ComparisonsVisualization, path_backup_csv_file
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest, \
    GevScaleChangePointTest, GevShapeChangePointTest


def visualize_all_stations():
    vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST, margin=150)
    vizu.visualize_maximum(visualize_metric_only=False)


def visualize_non_nan_station():
    for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][1:2]:
        vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST,
                                        keep_only_station_without_nan_values=True,
                                        normalize_observations=False,
                                        trend_test_class=trend_test_class)
        vizu.visualize_maximum(visualize_metric_only=True)
        # vizu.visualize_gev()


def example():
    # this is a really good example for the maxima at least
    # vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False)
    # vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Beaufortain', show=True)

    vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False, keep_only_station_without_nan_values=True)
    # vizu._visualize_ax_main(vizu.plot_gev, vizu.comparisons[0], 'Beaufortain', show=True)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Beaufortain', show=True)

def wrong_example():
    vizu = ComparisonsVisualization(altitudes=[1200], normalize_observations=False)
    # vizu._visualize_ax_main(vizu.plot_gev, vizu.comparisons[0], 'Beaufortain', show=True)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Chablais', show=True)

def wrong_example2():
    vizu = ComparisonsVisualization(altitudes=[1200], normalize_observations=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Vanoise', show=True)

    vizu = ComparisonsVisualization(altitudes=[1800], normalize_observations=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Haute-Maurienne', show=True)

    vizu = ComparisonsVisualization(altitudes=[600], normalize_observations=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Mercantour', show=True)

def wrong_example3():
    vizu = ComparisonsVisualization(altitudes=[1200], normalize_observations=False, keep_only_station_without_nan_values=True)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Devoluy', show=True)


def quick_metric_analysis():
    ComparisonsVisualization.visualize_metric(csv_filepath=path_backup_csv_file)
    # ComparisonsVisualization.visualize_metric()

if __name__ == '__main__':
    # wrong_example3()
    # visualize_fast_comparison()
    # visualize_all_stations()
    # quick_metric_analysis()
    # wrong_example2()
    visualize_non_nan_station()
    # example()

