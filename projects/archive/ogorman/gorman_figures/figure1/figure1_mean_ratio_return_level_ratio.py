from collections import OrderedDict

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall7Days, SafranSnowfall5Days
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import ALL_ALTITUDES_WITHOUT_NAN
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from projects.archive.ogorman.gorman_figures import \
    ComparativeCurveWrtAltitude

from projects.archive.ogorman.gorman_figures import \
    StudyVisualizerForReturnLevelChange


def load_altitude_to_study_visualizer(study_class, save_to_file=True) -> OrderedDict:
    altitude_to_study_visualizer = OrderedDict()
    for altitude in ALL_ALTITUDES_WITHOUT_NAN[2:10]:
    # for altitude in ALL_ALTITUDES_WITHOUT_NAN[2:5]:
        return_period = 30
        study_visualizer = StudyVisualizerForReturnLevelChange(study_class=study_class,
                                                               altitude=altitude,
                                                               return_period=return_period,
                                                               save_to_file=save_to_file,
                                                               fit_method=MarginFitMethod.extremes_fevd_l_moments)
        altitude_to_study_visualizer[altitude] = study_visualizer
    return altitude_to_study_visualizer


def plots():
    for study_class in [SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, SafranSnowfall7Days][:]:
        altitude_to_study_visualizer = load_altitude_to_study_visualizer(study_class, save_to_file=True)
        # for v in altitude_to_study_visualizer.values():
        #     v.all_plots()
        comparative_curve = ComparativeCurveWrtAltitude(altitude_to_study_visualizer)
        comparative_curve.all_plots()


if __name__ == '__main__':
    plots()
