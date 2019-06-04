from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST
from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis
from experiment.meteo_france_data.stations_data.visualization.comparisons_visualization.comparisons_visualization import \
    ComparisonsVisualization


def visualize_full_comparison():
    vizu = ComparisonsVisualization(altitudes=ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST, margin=150)
    vizu.visualize_maximum()


def visualize_fast_comparison():
    vizu = ComparisonsVisualization(altitudes=[900])
    vizu.visualize_maximum()


def example():
    vizu = ComparisonsVisualization(altitudes=[900])
    vizu._visualize_maximum(vizu.comparisons[0], 'Beaufortain', show=True)

if __name__ == '__main__':
    # visualize_fast_comparison()
    visualize_full_comparison()
    # example()

