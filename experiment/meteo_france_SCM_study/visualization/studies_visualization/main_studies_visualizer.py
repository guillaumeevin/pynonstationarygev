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
    study_iterator_global

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from collections import OrderedDict


def normal_visualization():
    for study_type in [ExtendedSafranTotalPrecip]:
        extended_studies = Studies(study_type)
        studies_visualizer = StudiesVisualizer(extended_studies)
        studies_visualizer.mean_as_a_function_of_altitude(region_only=True)


def altitude_trends():
    save_to_file = False
    only_first_one = False
    # altitudes that have 20 massifs at least
    altitudes = ALL_ALTITUDES[3:-6]
    # altitudes = ALL_ALTITUDES[:2]
    visualizers = [StudyVisualizer(study, save_to_file=save_to_file, temporal_non_stationarity=True, verbose=True,
                                   score_class=MedianScore)
                   for study in study_iterator_global(study_classes=[SafranSnowfall], only_first_one=only_first_one,
                                                      altitudes=altitudes)]
    altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
    visualizer = AltitudeVisualizer(altitude_to_visualizer)
    visualizer.negative_trend_percentages_evolution()


if __name__ == '__main__':
    altitude_trends()
