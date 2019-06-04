from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST
from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis
from experiment.meteo_france_data.stations_data.visualization.comparisons_visualization.comparisons_visualization import \
    ComparisonsVisualization


def visualize_all_stations():
    vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST, margin=150)
    vizu.visualize_maximum()


def visualize_non_nan_station():
    vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST,
                                    keep_only_station_without_nan_values=True,
                                    normalize_observations=True)
    # vizu.visualize_maximum()
    vizu.visualize_gev()


def example():
    # this is a really good example for the maxima at least
    vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Beaufortain', show=True)

    # vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False, keep_only_station_without_nan_values=True)
    # vizu._visualize_ax_main(vizu.plot_gev, vizu.comparisons[0], 'Beaufortain', show=True)

if __name__ == '__main__':
    # visualize_fast_comparison()
    # visualize_all_stations()
    visualize_non_nan_station()
    # example()

