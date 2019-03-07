from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import CrocusDepth, CrocusSwe, ExtendedCrocusDepth, \
    ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.safran.safran import Safran, ExtendedSafran
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies import Studies
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies_visualizer import StudiesVisualizer

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from collections import OrderedDict

SCM_STUDIES = [Safran, CrocusSwe, CrocusDepth]
SCM_EXTENDED_STUDIES = [ExtendedSafran, ExtendedCrocusSwe, ExtendedCrocusDepth]
SCM_STUDY_TO_EXTENDED_STUDY = OrderedDict(zip(SCM_STUDIES, SCM_EXTENDED_STUDIES))


def normal_visualization():
    for study_type in SCM_EXTENDED_STUDIES[1:2]:
        extended_studies = Studies(study_type)
        studies_visualizer = StudiesVisualizer(extended_studies)
        studies_visualizer.mean_as_a_function_of_altitude(region_only=True)


if __name__ == '__main__':
    normal_visualization()
