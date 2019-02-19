from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import CrocusDepth, CrocusSwe
from experiment.meteo_france_SCM_study.safran.safran import Safran
from itertools import product

from experiment.meteo_france_SCM_study.safran.safran_visualizer import StudyVisualizer


def load_all_studies(study_class, only_first_one=False):
    all_studies = []
    is_safran_study = study_class == Safran
    nb_days = [1, 5] if is_safran_study else [1]
    for alti, nb_day in product(AbstractStudy.ALTITUDES, nb_days):
        print('alti: {}, nb_day: {}'.format(alti, nb_day))
        study = Safran(alti, nb_day) if is_safran_study else study_class(alti)
        all_studies.append(study)
        if only_first_one:
            break
    return all_studies


if __name__ == '__main__':
    for study_class in [Safran, CrocusSwe, CrocusDepth][:]:
        for study in load_all_studies(study_class, only_first_one=True):
            study_visualizer = StudyVisualizer(study)
            # safran_visualizer.visualize_independent_margin_fits(threshold=[None, 20, 40, 60][0])
            study_visualizer.visualize_smooth_margin_fit()
            # safran_visualizer.visualize_full_fit()
