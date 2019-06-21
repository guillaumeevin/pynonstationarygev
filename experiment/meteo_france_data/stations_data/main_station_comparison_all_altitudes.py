from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST, ALL_ALTITUDES
from experiment.meteo_france_data.stations_data.visualization.comparisons_visualization.comparisons_visualization import \
    ComparisonsVisualization, path_backup_csv_file
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest, \
    GevScaleChangePointTest, GevShapeChangePointTest


# Create the map with the average error per massif

def visualize_all_stations_all_altitudes():
    vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES, margin=150, keep_only_station_without_nan_values=True)
    vizu.visualize_maximum(visualize_metric_only=False)


# Zoom on each massif

def example():
    vizu = ComparisonsVisualization(altitudes=[600], normalize_observations=False, keep_only_station_without_nan_values=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Mercantour', show=True, direct=True)
    # vizu = ComparisonsVisualization(altitudes=[300], normalize_observations=False, keep_only_station_without_nan_values=False)
    # vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Haut_Var-Haut_Verdon', show=True, direct=True)
    # vizu = ComparisonsVisualization(altitudes=[1800], normalize_observations=False, keep_only_station_without_nan_values=False)
    # vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Grandes-Rousses', show=True, direct=True)

def example_good():
    # vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False, keep_only_station_without_nan_values=False)
    # vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Beaufortain', show=True, direct=True)
    vizu = ComparisonsVisualization(altitudes=[900], normalize_observations=False, keep_only_station_without_nan_values=False)
    vizu._visualize_ax_main(vizu.plot_maxima, vizu.comparisons[0], 'Oisans', show=True, direct=True)

if __name__ == '__main__':
    # visualize_all_stations_all_altitudes()
    # example()
    example_good()

