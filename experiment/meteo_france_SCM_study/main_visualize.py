from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import CrocusDepth, CrocusSwe, ExtendedCrocusDepth, \
    ExtendedCrocusSwe
from experiment.meteo_france_SCM_study.safran.safran import Safran, ExtendedSafran
from itertools import product

from experiment.meteo_france_SCM_study.safran.safran_visualizer import StudyVisualizer
from collections import OrderedDict

SCM_STUDIES = [Safran, CrocusSwe, CrocusDepth]
SCM_EXTENDED_STUDIES = [ExtendedSafran, ExtendedCrocusSwe, ExtendedCrocusDepth]
SCM_STUDY_TO_EXTENDED_STUDY = OrderedDict(zip(SCM_STUDIES, SCM_EXTENDED_STUDIES))


def study_iterator(study_class, only_first_one=False, both_altitude=False, verbose=True):
    all_studies = []
    is_safran_study = study_class in [Safran, ExtendedSafran]
    nb_days = [3, 1] if is_safran_study else [1]
    if verbose:
        print('Loading studies....')
    for nb_day in nb_days:
        for alti in AbstractStudy.ALTITUDES[::1]:
            if verbose:
                print('alti: {}, nb_day: {}'.format(alti, nb_day))
            study = study_class(alti, nb_day) if is_safran_study else study_class(alti)
            yield study
            if only_first_one and not both_altitude:
                break
        if only_first_one:
            break

    return all_studies


def extended_visualization():
    for study_class in SCM_EXTENDED_STUDIES[1:2]:
        for study in study_iterator(study_class, only_first_one=True):
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_all_kde_graphs()


def normal_visualization():
    for study_class in SCM_STUDIES[:1]:
        for study in study_iterator(study_class, only_first_one=True):
            study_visualizer = StudyVisualizer(study)
            # study_visualizer.visualize_independent_margin_fits(threshold=[None, 20, 40, 60][0])
            study_visualizer.visualize_linear_margin_fit()


def complete_analysis(only_first_one=False):
    """An overview of everything that is possible with study OR extended study"""
    for study_class, extended_study_class in list(SCM_STUDY_TO_EXTENDED_STUDY.items())[:]:
        # First explore everything you can do with the extended study class
        print('Extended study')
        for extended_study in study_iterator(extended_study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(extended_study, save_to_file=True)
            study_visualizer.visualize_all_kde_graphs()
        print('Study normal')
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=True)
            study_visualizer.visualize_linear_margin_fit()


if __name__ == '__main__':
    # normal_visualization()
    extended_visualization()
    # complete_analysis()
