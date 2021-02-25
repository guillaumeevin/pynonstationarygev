from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_trend.trend_test.visualizers import \
    StudyVisualizerForShape
from extreme_trend.trend_test.visualizers import load_altitude_to_visualizer


def main_shape_repartition(altitudes, massif_names=None,
                           non_stationary_uncertainty=None, uncertainty_methods=None,
                           study_class=CrocusSnowLoadTotal,
                           study_visualizer_class=StudyVisualizerForShape, save_to_file=True):
    # Load altitude to visualizer
    altitude_to_visualizer = load_altitude_to_visualizer(altitudes, massif_names, non_stationary_uncertainty,
                                                         study_class, uncertainty_methods,
                                                         study_visualizer_class=study_visualizer_class,
                                                         save_to_file=save_to_file)
    altitudes_for_plot_trend = altitudes  #[900, 1800, 2700]
    visualizers_for_altitudes = [visualizer
                                 for altitude, visualizer in altitude_to_visualizer.items()
                                 if altitude in altitudes_for_plot_trend]
    max_abs_tdrl = max([visualizer.max_abs_change for visualizer in visualizers_for_altitudes])
    for visualizer in visualizers_for_altitudes:
        visualizer.plot_trends(max_abs_tdrl, add_colorbar=visualizer.study.altitude == 2700)
        # visualizer.plot_trends(max_abs_tdrl, add_colorbar=True)
        # visualizer.plot_trends()


if __name__ == '__main__':
    # main_shape_repartition([900], save_to_file=False)
    main_shape_repartition([900, 1800, 2700])
    # main_shape_repartition([300, 600, 900, 1200, 1500, 1800, 2700])
    # main_shape_repartition(paper_altitudes, study_visualizer_class=StudyVisualizerForShape, save_to_file=True)
    # main_shape_repartition([300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200],
    #                        study_visualizer_class=StudyVisualizerForShape, save_to_file=True)
