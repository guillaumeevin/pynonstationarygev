from experiment.meteo_france_SCM_study.abstract_score import MannKendall, WeigthedScore, MeanScore, MedianScore
from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import CrocusDepth, CrocusSwe, ExtendedCrocusDepth, \
    ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, \
    ExtendedSafranTotalPrecip
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies import Studies
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies_visualizer import StudiesVisualizer, \
    AltitudeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import ALL_ALTITUDES, \
    study_iterator_global, SCM_STUDIES

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from collections import OrderedDict


def normal_visualization():
    for study_type in [ExtendedSafranTotalPrecip]:
        extended_studies = Studies(study_type)
        studies_visualizer = StudiesVisualizer(extended_studies)
        studies_visualizer.mean_as_a_function_of_altitude(region_only=True)


def altitude_trends():
    save_to_file = True
    only_first_one = False
    # altitudes that have 20 massifs at least
    altitudes = ALL_ALTITUDES[3:-6]
    # altitudes = ALL_ALTITUDES[:2]
    for study_class in SCM_STUDIES[:]:
        for score_class in [MedianScore, MeanScore, MannKendall, WeigthedScore]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=True,
                                           score_class=score_class)
                           for study in study_iterator_global(study_classes=[study_class], only_first_one=only_first_one,
                                                              altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeVisualizer(altitude_to_visualizer, multiprocessing=False, save_to_file=save_to_file)
            visualizer.negative_trend_percentages_evolution(reverse=True)


if __name__ == '__main__':
    altitude_trends()
