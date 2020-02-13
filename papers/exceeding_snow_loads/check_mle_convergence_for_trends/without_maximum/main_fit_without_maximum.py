from typing import Dict

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from papers.exceeding_snow_loads.check_mle_convergence_for_trends.without_maximum.study_visualizer_for_fit_witout_maximum import \
    StudyVisualizerForFitWithoutMaximum


def fit_without_maximum_value(altitude_to_visualizer: Dict[int, StudyVisualizerForFitWithoutMaximum]):
    for v in altitude_to_visualizer.values():
        v.maximum_value_test()


if __name__ == '__main__':
    altitudes = ALL_ALTITUDES_WITHOUT_NAN[:]
    altitude_to_visualizer = {altitude: StudyVisualizerForFitWithoutMaximum(CrocusSnowLoadTotal(altitude=altitude),
                                                                            multiprocessing=True)
                              for altitude in altitudes}
    fit_without_maximum_value(altitude_to_visualizer)
