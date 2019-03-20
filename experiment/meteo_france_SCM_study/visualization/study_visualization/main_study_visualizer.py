from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus import CrocusDepth, CrocusSwe, ExtendedCrocusDepth, \
    ExtendedCrocusSwe, CrocusDaysWithSnowOnGround
from experiment.meteo_france_SCM_study.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, SafranRainfall, \
    SafranTemperature, SafranTotalPrecip

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from collections import OrderedDict

SCM_STUDIES = [SafranSnowfall, CrocusSwe, CrocusDepth]
SCM_EXTENDED_STUDIES = [ExtendedSafranSnowfall, ExtendedCrocusSwe, ExtendedCrocusDepth]
SCM_STUDY_TO_EXTENDED_STUDY = OrderedDict(zip(SCM_STUDIES, SCM_EXTENDED_STUDIES))


def study_iterator(study_class, only_first_one=False, both_altitude=False, verbose=True):
    all_studies = []
    is_safran_study = study_class in [SafranSnowfall, ExtendedSafranSnowfall]
    nb_days = [1] if is_safran_study else [1]
    if verbose:
        print('Loading studies....')
    for nb_day in nb_days:
        for alti in AbstractStudy.ALTITUDES[::1]:
            if verbose:
                print('alti: {}, nb_day: {}'.format(alti, nb_day))
            study = study_class(altitude=alti, nb_consecutive_days=nb_day) if is_safran_study \
                else study_class(altitude=alti)
            yield study
            if only_first_one and not both_altitude:
                break
        if only_first_one:
            break

    return all_studies


def extended_visualization():
    save_to_file = False
    only_first_one = True
    for study_class in SCM_EXTENDED_STUDIES[-1:]:
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file, only_one_graph=True,
                                               plot_block_maxima_quantiles=False)
            # study_visualizer.visualize_all_mean_and_max_graphs()
            study_visualizer.visualize_all_experimental_law()
    # for study_class in SCM_EXTENDED_STUDIES[:]:
    #     for study in study_iterator(study_class, only_first_one=False):
    #         study_visualizer = StudyVisualizer(study, single_massif_graph=True, save_to_file=True)
    #         # study_visualizer.visualize_all_kde_graphs()
    #         study_visualizer.visualize_all_experimental_law()


def annual_mean_vizu_compare_durand_study(safran=True, take_mean_value=True, altitude=1800):
    if safran:
        for study_class in [SafranTotalPrecip, SafranRainfall, SafranSnowfall, SafranTemperature][2:3]:
            study = study_class(altitude=altitude, year_min=1958, year_max=2002)
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_annual_mean_values(take_mean_value=True)
    else:
        for study_class in [CrocusSwe, CrocusDepth, CrocusDaysWithSnowOnGround][-1:]:
            study = study_class(altitude=altitude, year_min=1958, year_max=2005)
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_annual_mean_values(take_mean_value=take_mean_value)


def max_stable_process_vizu_compare_gaume_study(altitude=1800, nb_days=1):
    study = SafranSnowfall(altitude=altitude, nb_consecutive_days=nb_days)
    study_visualizer = StudyVisualizer(study)
    study_visualizer.visualize_brown_resnick_fit()


def normal_visualization():
    save_to_file = False
    only_first_one = True
    # for study_class in SCM_STUDIES[:1]:
    for study_class in [SafranSnowfall, SafranRainfall, SafranTemperature][:1]:
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file)
            # study_visualizer.visualize_independent_margin_fits(threshold=[None, 20, 40, 60][0])
            # study_visualizer.visualize_annual_mean_values()
            study_visualizer.visualize_linear_margin_fit(only_first_max_stable=True)


def complete_analysis(only_first_one=False):
    """An overview of everything that is possible with study OR extended study"""
    for study_class, extended_study_class in list(SCM_STUDY_TO_EXTENDED_STUDY.items())[:]:
        # First explore everything you can do with the extended study class
        print('Extended study')
        for extended_study in study_iterator(extended_study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(extended_study, save_to_file=True)
            study_visualizer.visualize_all_mean_and_max_graphs()
            study_visualizer.visualize_all_experimental_law()
        print('Study normal')
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=True)
            study_visualizer.visualize_linear_margin_fit()


if __name__ == '__main__':
    # annual_mean_vizu_compare_durand_study(safran=True, take_mean_value=True, altitude=2400)
    normal_visualization()
    # max_stable_process_vizu_compare_gaume_study(altitude=1800, nb_days=1)
    # extended_visualization()
    # complete_analysis()
