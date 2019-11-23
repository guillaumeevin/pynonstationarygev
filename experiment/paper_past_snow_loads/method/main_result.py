from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.paper_past_snow_loads.method.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends

ALTITUDE_TRENDS = [1800]


def draw_snow_load_map(altitude):
    study = CrocusSnowLoadTotal(altitude=altitude)
    visualizer = StudyVisualizerForNonStationaryTrends(study, multiprocessing=True)
    visualizer.plot_trends()

if __name__ == '__main__':
    draw_snow_load_map(altitude=1800)

